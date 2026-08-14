#include "ed_cudnn_conv_transpose_1d.h"

#include <cuda_runtime.h>
#include <cudnn.h>

#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
#include <unordered_map>

namespace {

constexpr size_t ED_CUDNN_CONV_TRANSPOSE_1D_MAX_WORKSPACE = 1152ull * 1024ull * 1024ull;

struct workspace_key {
    int device;
    cudaStream_t stream;
    bool operator==(const workspace_key & other) const { return device == other.device && stream == other.stream; }
};

struct workspace_key_hash {
    size_t operator()(const workspace_key & key) const {
        return std::hash<int>()(key.device) ^ (std::hash<uintptr_t>()((uintptr_t) key.stream) << 1);
    }
};

struct workspace {
    int device = 0;
    void * ptr = nullptr;
    size_t size = 0;
    ~workspace() {
        if (ptr != nullptr) {
            int previous_device = 0;
            cudaGetDevice(&previous_device);
            cudaSetDevice(device);
            cudaFree(ptr);
            cudaSetDevice(previous_device);
        }
    }
};

std::mutex g_workspace_mutex;
std::unordered_map<workspace_key, std::unique_ptr<workspace>, workspace_key_hash> g_workspaces;

static bool explicitly_enabled() {
    static const bool value = [] {
        const char * env = getenv("ED_CUDNN_CONV_TRANSPOSE_1D");
        return env != nullptr && env[0] != '\0' && atoi(env) != 0;
    }();
    return value;
}

static bool enabled(const ggml_tensor * dst) {
    return explicitly_enabled() ||
           (dst != nullptr &&
            std::strcmp(dst->name, "minimax_h3.audio_vae.conv_transpose_1d") == 0);
}

static bool contiguous(const ggml_tensor * tensor) {
    return ggml_is_contiguous(tensor);
}

static bool supported(const ggml_tensor * dst, int & stride) {
    if (!enabled(dst) || dst == nullptr || dst->op != GGML_OP_CONV_TRANSPOSE_1D ||
        dst->src[0] == nullptr || dst->src[1] == nullptr) {
        return false;
    }
    const ggml_tensor * weight = dst->src[0];
    const ggml_tensor * input = dst->src[1];
    if (weight->type != GGML_TYPE_F32 || input->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32 ||
        !contiguous(weight) || !contiguous(input) || !contiguous(dst) ||
        weight->ne[3] != 1 || input->ne[3] != 1 || dst->ne[3] != 1 ||
        input->ne[1] != weight->ne[2] || dst->ne[1] != weight->ne[1] ||
        input->ne[2] != dst->ne[2]) {
        return false;
    }
    const int32_t * params = reinterpret_cast<const int32_t *>(dst->op_params);
    stride = params[0];
    return stride > 1 && params[1] == 0 && params[2] == 1 &&
           dst->ne[0] == (input->ne[0] - 1) * stride + weight->ne[0];
}

static void * get_workspace(int device, cudaStream_t stream, size_t required) {
    if (required == 0) {
        return nullptr;
    }
    std::lock_guard<std::mutex> lock(g_workspace_mutex);
    workspace_key key{device, stream};
    auto & entry = g_workspaces[key];
    if (!entry) {
        entry = std::make_unique<workspace>();
        entry->device = device;
    }
    if (entry->size >= required) {
        return entry->ptr;
    }
    void * replacement = nullptr;
    if (cudaMalloc(&replacement, required) != cudaSuccess) {
        return nullptr;
    }
    cudaFree(entry->ptr);
    entry->ptr = replacement;
    entry->size = required;
    return replacement;
}

} // namespace

ed_cudnn_conv_transpose_1d_result_t ed_cudnn_conv_transpose_1d_compute(
        ggml_tensor * dst, ed_cudnn_conv_transpose_1d_stream_t stream_ptr) {
    int stride = 0;
    if (!supported(dst, stride)) {
        return ED_CUDNN_CONV_TRANSPOSE_1D_UNSUPPORTED;
    }

    const ggml_tensor * weight = dst->src[0];
    const ggml_tensor * input = dst->src[1];
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);
    int device = 0;
    if (cudaGetDevice(&device) != cudaSuccess) {
        return ED_CUDNN_CONV_TRANSPOSE_1D_BUILD_FAILED;
    }

    cudnnHandle_t handle = nullptr;
    cudnnTensorDescriptor_t input_desc = nullptr;
    cudnnTensorDescriptor_t output_desc = nullptr;
    cudnnFilterDescriptor_t filter_desc = nullptr;
    cudnnConvolutionDescriptor_t conv_desc = nullptr;
    auto cleanup = [&] {
        if (conv_desc) cudnnDestroyConvolutionDescriptor(conv_desc);
        if (filter_desc) cudnnDestroyFilterDescriptor(filter_desc);
        if (output_desc) cudnnDestroyTensorDescriptor(output_desc);
        if (input_desc) cudnnDestroyTensorDescriptor(input_desc);
        if (handle) cudnnDestroy(handle);
    };
    if (cudnnCreate(&handle) != CUDNN_STATUS_SUCCESS || cudnnSetStream(handle, stream) != CUDNN_STATUS_SUCCESS ||
        cudnnCreateTensorDescriptor(&input_desc) != CUDNN_STATUS_SUCCESS ||
        cudnnCreateTensorDescriptor(&output_desc) != CUDNN_STATUS_SUCCESS ||
        cudnnCreateFilterDescriptor(&filter_desc) != CUDNN_STATUS_SUCCESS ||
        cudnnCreateConvolutionDescriptor(&conv_desc) != CUDNN_STATUS_SUCCESS) {
        cleanup();
        return ED_CUDNN_CONV_TRANSPOSE_1D_BUILD_FAILED;
    }

    const int n = (int) input->ne[2];
    const int input_channels = (int) input->ne[1];
    const int output_channels = (int) dst->ne[1];
    const int input_width = (int) input->ne[0];
    const int output_width = (int) dst->ne[0];
    const int kernel_width = (int) weight->ne[0];
    const int filter_dims[] = {input_channels, output_channels, 1, kernel_width};
    const int pad[] = {0, 0};
    const int strides[] = {1, stride};
    const int dilation[] = {1, 1};
    if (cudnnSetTensor4dDescriptor(input_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                                   n, input_channels, 1, input_width) != CUDNN_STATUS_SUCCESS ||
        cudnnSetTensor4dDescriptor(output_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                                   n, output_channels, 1, output_width) != CUDNN_STATUS_SUCCESS ||
        cudnnSetFilterNdDescriptor(filter_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, 4, filter_dims) != CUDNN_STATUS_SUCCESS ||
        cudnnSetConvolutionNdDescriptor(conv_desc, 2, pad, strides, dilation, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT) != CUDNN_STATUS_SUCCESS) {
        cleanup();
        return ED_CUDNN_CONV_TRANSPOSE_1D_BUILD_FAILED;
    }

    int returned = 0;
    cudnnConvolutionBwdDataAlgoPerf_t perf{};
    if (cudnnGetConvolutionBackwardDataAlgorithm_v7(handle, filter_desc, input_desc, conv_desc, output_desc, 1, &returned, &perf) != CUDNN_STATUS_SUCCESS ||
        returned == 0 || perf.status != CUDNN_STATUS_SUCCESS || perf.memory > ED_CUDNN_CONV_TRANSPOSE_1D_MAX_WORKSPACE) {
        cleanup();
        return ED_CUDNN_CONV_TRANSPOSE_1D_BUILD_FAILED;
    }
    void * scratch = get_workspace(device, stream, perf.memory);
    if (perf.memory != 0 && scratch == nullptr) {
        cleanup();
        return ED_CUDNN_CONV_TRANSPOSE_1D_BUILD_FAILED;
    }
    const float alpha = 1.0f;
    const float beta = 0.0f;
    const cudnnStatus_t status = cudnnConvolutionBackwardData(handle, &alpha, filter_desc, weight->data,
                                                               input_desc, input->data, conv_desc, perf.algo,
                                                               scratch, perf.memory, &beta, output_desc, dst->data);
    cleanup();
    return status == CUDNN_STATUS_SUCCESS ? ED_CUDNN_CONV_TRANSPOSE_1D_SUCCESS : ED_CUDNN_CONV_TRANSPOSE_1D_EXECUTE_FAILED;
}
