#include "ed_async_offload.h"

#include "common.cuh"             // ggml_backend_cuda_context (for compute_wait)
#include "ggml-backend-impl.h"    // complete struct ggml_backend (backend->context)

#include <cuda_runtime.h>

static inline void ed_async_offload_check(cudaError_t err, const char * what) {
    if (err != cudaSuccess) {
        GGML_LOG_ERROR("ed_async_offload: %s failed: %s\n", what, cudaGetErrorString(err));
        GGML_ABORT("ed_async_offload fatal error");
    }
}

ed_copy_stream_t ed_async_offload_stream_create(int device) {
    int prev = 0;
    ed_async_offload_check(cudaGetDevice(&prev), "cudaGetDevice");
    ed_async_offload_check(cudaSetDevice(device), "cudaSetDevice");
    cudaStream_t stream = nullptr;
    ed_async_offload_check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "cudaStreamCreateWithFlags");
    ed_async_offload_check(cudaSetDevice(prev), "cudaSetDevice(restore)");
    return reinterpret_cast<ed_copy_stream_t>(stream);
}

void ed_async_offload_stream_destroy(ed_copy_stream_t stream) {
    if (stream == nullptr) {
        return;
    }
    cudaStreamDestroy(reinterpret_cast<cudaStream_t>(stream));
}

ed_copy_event_t ed_async_offload_event_create() {
    cudaEvent_t event = nullptr;
    ed_async_offload_check(cudaEventCreateWithFlags(&event, cudaEventDisableTiming), "cudaEventCreateWithFlags");
    return reinterpret_cast<ed_copy_event_t>(event);
}

void ed_async_offload_event_destroy(ed_copy_event_t event) {
    if (event == nullptr) {
        return;
    }
    cudaEventDestroy(reinterpret_cast<cudaEvent_t>(event));
}

void ed_async_offload_h2d(void * dst, const void * src, size_t nbytes, ed_copy_stream_t stream) {
    ed_async_offload_check(
        cudaMemcpyAsync(dst, src, nbytes, cudaMemcpyHostToDevice, reinterpret_cast<cudaStream_t>(stream)),
        "cudaMemcpyAsync H2D");
}

void ed_async_offload_event_record(ed_copy_event_t event, ed_copy_stream_t stream) {
    ed_async_offload_check(
        cudaEventRecord(reinterpret_cast<cudaEvent_t>(event), reinterpret_cast<cudaStream_t>(stream)),
        "cudaEventRecord");
}

void ed_async_offload_event_synchronize(ed_copy_event_t event) {
    ed_async_offload_check(cudaEventSynchronize(reinterpret_cast<cudaEvent_t>(event)), "cudaEventSynchronize");
}

void ed_async_offload_compute_wait(ggml_backend_t compute_backend, ed_copy_event_t event) {
    ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *) compute_backend->context;
    ed_async_offload_check(
        cudaStreamWaitEvent(cuda_ctx->stream(), reinterpret_cast<cudaEvent_t>(event), 0),
        "cudaStreamWaitEvent");
}
