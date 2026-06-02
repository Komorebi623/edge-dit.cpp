#include "backend/ggml/parallel/ggml_comm.hpp"

#include <sstream>
#include <stdexcept>
#include <string>

namespace edgedit::ggml_comm {
namespace {

void set_error(std::string* error, const std::string& message) {
    if (error != nullptr) {
        *error = message;
    }
}

std::string tensor_desc(const ggml_tensor* tensor) {
    if (tensor == nullptr) {
        return "<null>";
    }

    std::ostringstream oss;
    oss << (tensor->name[0] != '\0' ? tensor->name : "<unnamed>")
        << " type=" << ggml_type_name(tensor->type)
        << " ne=[" << tensor->ne[0]
        << "," << tensor->ne[1]
        << "," << tensor->ne[2]
        << "," << tensor->ne[3]
        << "]"
        << " op=" << ggml_op_name(tensor->op);
    return oss.str();
}

DataType dtype_or_throw(ggml_type type) {
    DataType dtype;
    std::string error;
    if (!ggml_type_to_comm_dtype(type, &dtype, &error)) {
        throw std::invalid_argument(error);
    }
    return dtype;
}

Buffer tensor_to_buffer_or_throw(ggml_tensor* tensor) {
    Buffer buffer;
    std::string error;
    if (!tensor_to_buffer(tensor, &buffer, &error)) {
        throw std::invalid_argument(error);
    }
    return buffer;
}

void check_same_dtype_or_throw(const ggml_tensor* input,
                               const ggml_tensor* output,
                               const char* op_name) {
    if (!same_dtype(input, output)) {
        std::ostringstream oss;
        oss << op_name << " input/output dtype mismatch: input="
            << tensor_desc(input) << " output=" << tensor_desc(output);
        throw std::invalid_argument(oss.str());
    }
}

void check_same_numel_or_throw(const ggml_tensor* input,
                               const ggml_tensor* output,
                               const char* op_name) {
    if (!same_numel(input, output)) {
        std::ostringstream oss;
        oss << op_name << " input/output numel mismatch: input_numel="
            << tensor_numel(input) << " output_numel=" << tensor_numel(output)
            << " input=" << tensor_desc(input)
            << " output=" << tensor_desc(output);
        throw std::invalid_argument(oss.str());
    }
}

void check_all_gather_shape_or_throw(ProcessGroup& group,
                                     const ggml_tensor* input,
                                     const ggml_tensor* output) {
    const size_t input_count = tensor_numel(input);
    const size_t output_count = tensor_numel(output);
    const size_t expected = input_count * static_cast<size_t>(group.size());

    if (output_count != expected) {
        std::ostringstream oss;
        oss << "all_gather output numel must equal input_numel * world_size: "
            << "input_numel=" << input_count
            << " world_size=" << group.size()
            << " expected_output_numel=" << expected
            << " actual_output_numel=" << output_count
            << " input=" << tensor_desc(input)
            << " output=" << tensor_desc(output);
        throw std::invalid_argument(oss.str());
    }
}

void check_all_to_all_shape_or_throw(ProcessGroup& group,
                                     const ggml_tensor* input,
                                     const ggml_tensor* output,
                                     size_t count_per_peer) {
    const size_t expected = count_per_peer * static_cast<size_t>(group.size());
    const size_t input_count = tensor_numel(input);
    const size_t output_count = tensor_numel(output);

    if (input_count != expected || output_count != expected) {
        std::ostringstream oss;
        oss << "all_to_all input/output numel must equal count_per_peer * world_size: "
            << "count_per_peer=" << count_per_peer
            << " world_size=" << group.size()
            << " expected_numel=" << expected
            << " input_numel=" << input_count
            << " output_numel=" << output_count
            << " input=" << tensor_desc(input)
            << " output=" << tensor_desc(output);
        throw std::invalid_argument(oss.str());
    }
}

} // namespace

// ============================================================
// Basic tensor inspection / conversion helpers
// ============================================================

bool ggml_type_to_comm_dtype(
    ggml_type type,
    DataType* dtype_out,
    std::string* error
) {
    if (dtype_out == nullptr) {
        set_error(error, "ggml_type_to_comm_dtype dtype_out is null");
        return false;
    }

    switch (type) {
        case GGML_TYPE_F32:
            *dtype_out = DataType::kFloat32;
            return true;
        case GGML_TYPE_F16:
            *dtype_out = DataType::kFloat16;
            return true;
        case GGML_TYPE_BF16:
            *dtype_out = DataType::kBFloat16;
            return true;
        case GGML_TYPE_I32:
            *dtype_out = DataType::kInt32;
            return true;
        case GGML_TYPE_I64:
            *dtype_out = DataType::kInt64;
            return true;
        default: {
            std::ostringstream oss;
            oss << "unsupported ggml dtype for communication: "
                << ggml_type_name(type)
                << ". Phase 1 only supports F32, F16, BF16, I32 and I64.";
            set_error(error, oss.str());
            return false;
        }
    }
}

bool get_tensor_comm_info(
    ggml_tensor* tensor,
    TensorCommInfo* info_out,
    std::string* error
) {
    if (info_out == nullptr) {
        set_error(error, "get_tensor_comm_info info_out is null");
        return false;
    }

    *info_out = TensorCommInfo{};

    if (tensor == nullptr) {
        set_error(error, "get_tensor_comm_info tensor is null");
        return false;
    }

    DataType dtype;
    if (!ggml_type_to_comm_dtype(tensor->type, &dtype, error)) {
        return false;
    }

    ggml_backend_buffer_t buffer =
        tensor->view_src != nullptr ? tensor->view_src->buffer : tensor->buffer;

    info_out->tensor = tensor;
    info_out->buffer = buffer;
    info_out->data = tensor->data;
    info_out->count = static_cast<size_t>(ggml_nelements(tensor));
    info_out->dtype = dtype;
    info_out->contiguous = ggml_is_contiguous(tensor);
    info_out->has_backend_buffer = buffer != nullptr;
    info_out->has_data = tensor->data != nullptr;

    return true;
}

bool tensor_to_buffer(
    ggml_tensor* tensor,
    Buffer* buffer_out,
    std::string* error
) {
    if (buffer_out == nullptr) {
        set_error(error, "tensor_to_buffer buffer_out is null");
        return false;
    }

    *buffer_out = Buffer{};

    TensorCommInfo info;
    if (!get_tensor_comm_info(tensor, &info, error)) {
        return false;
    }

    if (!info.contiguous) {
        std::ostringstream oss;
        oss << "communication requires contiguous ggml tensor in phase 1: "
            << tensor_desc(tensor);
        set_error(error, oss.str());
        return false;
    }

    if (!info.has_backend_buffer) {
        std::ostringstream oss;
        oss << "ggml tensor has no backend buffer: "
            << tensor_desc(tensor);
        set_error(error, oss.str());
        return false;
    }

    if (!info.has_data) {
        std::ostringstream oss;
        oss << "ggml tensor data pointer is null: "
            << tensor_desc(tensor);
        set_error(error, oss.str());
        return false;
    }

    buffer_out->data = info.data;
    buffer_out->count = info.count;
    buffer_out->type = info.dtype;
    buffer_out->device = -1;

    return true;
}

bool same_dtype(
    const ggml_tensor* a,
    const ggml_tensor* b
) {
    if (a == nullptr || b == nullptr) {
        return false;
    }
    return a->type == b->type;
}

bool same_numel(
    const ggml_tensor* a,
    const ggml_tensor* b
) {
    if (a == nullptr || b == nullptr) {
        return false;
    }
    return ggml_nelements(a) == ggml_nelements(b);
}

size_t tensor_numel(
    const ggml_tensor* tensor
) {
    if (tensor == nullptr) {
        return 0;
    }
    return static_cast<size_t>(ggml_nelements(tensor));
}

// ============================================================
// Sync collective APIs using ProcessGroup directly
// ============================================================

bool all_reduce(
    ProcessGroup& group,
    ggml_tensor* input,
    ggml_tensor* output,
    ReduceOp op,
    std::string* error
) {
    try {
        check_same_dtype_or_throw(input, output, "all_reduce");
        check_same_numel_or_throw(input, output, "all_reduce");

        Buffer input_buffer = tensor_to_buffer_or_throw(input);
        Buffer output_buffer = tensor_to_buffer_or_throw(output);

        group.all_reduce(input_buffer, output_buffer, op);
        return true;
    } catch (const std::exception& e) {
        set_error(error, e.what());
        return false;
    }
}

bool all_gather(
    ProcessGroup& group,
    ggml_tensor* input,
    ggml_tensor* output,
    std::string* error
) {
    try {
        check_same_dtype_or_throw(input, output, "all_gather");
        check_all_gather_shape_or_throw(group, input, output);

        Buffer input_buffer = tensor_to_buffer_or_throw(input);
        Buffer output_buffer = tensor_to_buffer_or_throw(output);

        group.all_gather(input_buffer, output_buffer);
        return true;
    } catch (const std::exception& e) {
        set_error(error, e.what());
        return false;
    }
}

bool all_to_all(
    ProcessGroup& group,
    ggml_tensor* input,
    ggml_tensor* output,
    size_t count_per_peer,
    std::string* error
) {
    try {
        check_same_dtype_or_throw(input, output, "all_to_all");
        check_all_to_all_shape_or_throw(group, input, output, count_per_peer);

        Buffer input_buffer = tensor_to_buffer_or_throw(input);
        Buffer output_buffer = tensor_to_buffer_or_throw(output);

        group.all_to_all(input_buffer, output_buffer, count_per_peer);
        return true;
    } catch (const std::exception& e) {
        set_error(error, e.what());
        return false;
    }
}

bool broadcast(
    ProcessGroup& group,
    ggml_tensor* tensor,
    int root,
    std::string* error
) {
    try {
        if (root < 0 || root >= group.size()) {
            std::ostringstream oss;
            oss << "broadcast root out of range: root=" << root
                << " world_size=" << group.size();
            throw std::invalid_argument(oss.str());
        }

        Buffer buffer = tensor_to_buffer_or_throw(tensor);
        group.broadcast(buffer, root);
        return true;
    } catch (const std::exception& e) {
        set_error(error, e.what());
        return false;
    }
}

// ============================================================
// Sync collective APIs using ParallelContext
// ============================================================

bool all_reduce(
    ParallelContext& parallel,
    ggml_tensor* input,
    ggml_tensor* output,
    ReduceOp op,
    std::string* error
) {
    return all_reduce(parallel.world_group(), input, output, op, error);
}

bool all_gather(
    ParallelContext& parallel,
    ggml_tensor* input,
    ggml_tensor* output,
    std::string* error
) {
    return all_gather(parallel.world_group(), input, output, error);
}

bool all_to_all(
    ParallelContext& parallel,
    ggml_tensor* input,
    ggml_tensor* output,
    size_t count_per_peer,
    std::string* error
) {
    return all_to_all(parallel.world_group(), input, output, count_per_peer, error);
}

bool broadcast(
    ParallelContext& parallel,
    ggml_tensor* tensor,
    int root,
    std::string* error
) {
    return broadcast(parallel.world_group(), tensor, root, error);
}

// ============================================================
// Async collective APIs using ProcessGroup directly
// ============================================================

std::unique_ptr<Work> all_reduce_async(
    ProcessGroup& group,
    ggml_tensor* input,
    ggml_tensor* output,
    ReduceOp op
) {
    check_same_dtype_or_throw(input, output, "all_reduce_async");
    check_same_numel_or_throw(input, output, "all_reduce_async");

    Buffer input_buffer = tensor_to_buffer_or_throw(input);
    Buffer output_buffer = tensor_to_buffer_or_throw(output);

    return group.all_reduce_async(input_buffer, output_buffer, op);
}

std::unique_ptr<Work> all_gather_async(
    ProcessGroup& group,
    ggml_tensor* input,
    ggml_tensor* output
) {
    check_same_dtype_or_throw(input, output, "all_gather_async");
    check_all_gather_shape_or_throw(group, input, output);

    Buffer input_buffer = tensor_to_buffer_or_throw(input);
    Buffer output_buffer = tensor_to_buffer_or_throw(output);

    return group.all_gather_async(input_buffer, output_buffer);
}

std::unique_ptr<Work> all_to_all_async(
    ProcessGroup& group,
    ggml_tensor* input,
    ggml_tensor* output,
    size_t count_per_peer
) {
    check_same_dtype_or_throw(input, output, "all_to_all_async");
    check_all_to_all_shape_or_throw(group, input, output, count_per_peer);

    Buffer input_buffer = tensor_to_buffer_or_throw(input);
    Buffer output_buffer = tensor_to_buffer_or_throw(output);

    return group.all_to_all_async(input_buffer, output_buffer, count_per_peer);
}

std::unique_ptr<Work> broadcast_async(
    ProcessGroup& group,
    ggml_tensor* tensor,
    int root
) {
    if (root < 0 || root >= group.size()) {
        std::ostringstream oss;
        oss << "broadcast_async root out of range: root=" << root
            << " world_size=" << group.size();
        throw std::invalid_argument(oss.str());
    }

    Buffer buffer = tensor_to_buffer_or_throw(tensor);
    return group.broadcast_async(buffer, root);
}

// ============================================================
// Async collective APIs using ParallelContext
// ============================================================

std::unique_ptr<Work> all_reduce_async(
    ParallelContext& parallel,
    ggml_tensor* input,
    ggml_tensor* output,
    ReduceOp op
) {
    return all_reduce_async(parallel.world_group(), input, output, op);
}

std::unique_ptr<Work> all_gather_async(
    ParallelContext& parallel,
    ggml_tensor* input,
    ggml_tensor* output
) {
    return all_gather_async(parallel.world_group(), input, output);
}

std::unique_ptr<Work> all_to_all_async(
    ParallelContext& parallel,
    ggml_tensor* input,
    ggml_tensor* output,
    size_t count_per_peer
) {
    return all_to_all_async(parallel.world_group(), input, output, count_per_peer);
}

std::unique_ptr<Work> broadcast_async(
    ParallelContext& parallel,
    ggml_tensor* tensor,
    int root
) {
    return broadcast_async(parallel.world_group(), tensor, root);
}

} // namespace edgedit::ggml_comm