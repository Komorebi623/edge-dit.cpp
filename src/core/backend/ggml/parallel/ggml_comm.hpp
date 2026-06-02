#ifndef __ED_GGML_COMM_HPP__
#define __ED_GGML_COMM_HPP__

#include <cstddef>
#include <memory>
#include <string>

#include "ggml.h"
#include "ggml-backend.h"

#include "core/parallel/parallel_context.hpp"
#include "core/parallel/process_group.hpp"

namespace edgedit::ggml_comm {

using edgedit::parallel::Buffer;
using edgedit::parallel::DataType;
using edgedit::parallel::ParallelContext;
using edgedit::parallel::ProcessGroup;
using edgedit::parallel::ReduceOp;
using edgedit::parallel::Work;

// ============================================================
// Basic tensor inspection / conversion helpers
// ============================================================

struct TensorCommInfo {
    ggml_tensor* tensor              = nullptr;
    ggml_backend_buffer_t buffer     = nullptr;
    void* data                       = nullptr;
    size_t count                     = 0;
    DataType dtype                   = DataType::kFloat32;
    bool contiguous                  = false;
    bool has_backend_buffer          = false;
    bool has_data                    = false;
};

// Convert ggml_type to ProcessGroup DataType.
// Phase 1 supports only plain dense communication dtypes.
// Quantized tensors are intentionally unsupported for now.
bool ggml_type_to_comm_dtype(
    ggml_type type,
    DataType* dtype_out,
    std::string* error = nullptr
);

// Inspect a ggml tensor and produce TensorCommInfo.
// Does not launch communication.
bool get_tensor_comm_info(
    ggml_tensor* tensor,
    TensorCommInfo* info_out,
    std::string* error = nullptr
);

// Convert a contiguous ggml_tensor to ProcessGroup::Buffer.
// Requirements:
// - tensor != nullptr
// - tensor is contiguous
// - tensor has backend buffer
// - tensor has non-null data pointer
// - tensor type is supported by ggml_type_to_comm_dtype
bool tensor_to_buffer(
    ggml_tensor* tensor,
    Buffer* buffer_out,
    std::string* error = nullptr
);

// Utility shape/count validators.
bool same_dtype(
    const ggml_tensor* a,
    const ggml_tensor* b
);

bool same_numel(
    const ggml_tensor* a,
    const ggml_tensor* b
);

size_t tensor_numel(
    const ggml_tensor* tensor
);

// ============================================================
// Sync collective APIs using ProcessGroup directly
// ============================================================

bool all_reduce(
    ProcessGroup& group,
    ggml_tensor* input,
    ggml_tensor* output,
    ReduceOp op = ReduceOp::kSum,
    std::string* error = nullptr
);

bool all_gather(
    ProcessGroup& group,
    ggml_tensor* input,
    ggml_tensor* output,
    std::string* error = nullptr
);

bool all_to_all(
    ProcessGroup& group,
    ggml_tensor* input,
    ggml_tensor* output,
    size_t count_per_peer,
    std::string* error = nullptr
);

bool broadcast(
    ProcessGroup& group,
    ggml_tensor* tensor,
    int root,
    std::string* error = nullptr
);

// ============================================================
// Sync collective APIs using ParallelContext
// These simply dispatch to parallel.world_group().
// ============================================================

bool all_reduce(
    ParallelContext& parallel,
    ggml_tensor* input,
    ggml_tensor* output,
    ReduceOp op = ReduceOp::kSum,
    std::string* error = nullptr
);

bool all_gather(
    ParallelContext& parallel,
    ggml_tensor* input,
    ggml_tensor* output,
    std::string* error = nullptr
);

bool all_to_all(
    ParallelContext& parallel,
    ggml_tensor* input,
    ggml_tensor* output,
    size_t count_per_peer,
    std::string* error = nullptr
);

bool broadcast(
    ParallelContext& parallel,
    ggml_tensor* tensor,
    int root,
    std::string* error = nullptr
);

// ============================================================
// Async collective APIs using ProcessGroup directly
//
// These return Work from the underlying ProcessGroup.
// The caller is responsible for calling work->wait()
// before consuming the output tensor.
// ============================================================

std::unique_ptr<Work> all_reduce_async(
    ProcessGroup& group,
    ggml_tensor* input,
    ggml_tensor* output,
    ReduceOp op = ReduceOp::kSum
);

std::unique_ptr<Work> all_gather_async(
    ProcessGroup& group,
    ggml_tensor* input,
    ggml_tensor* output
);

std::unique_ptr<Work> all_to_all_async(
    ProcessGroup& group,
    ggml_tensor* input,
    ggml_tensor* output,
    size_t count_per_peer
);

std::unique_ptr<Work> broadcast_async(
    ProcessGroup& group,
    ggml_tensor* tensor,
    int root
);

// ============================================================
// Async collective APIs using ParallelContext
// ============================================================

std::unique_ptr<Work> all_reduce_async(
    ParallelContext& parallel,
    ggml_tensor* input,
    ggml_tensor* output,
    ReduceOp op = ReduceOp::kSum
);

std::unique_ptr<Work> all_gather_async(
    ParallelContext& parallel,
    ggml_tensor* input,
    ggml_tensor* output
);

std::unique_ptr<Work> all_to_all_async(
    ParallelContext& parallel,
    ggml_tensor* input,
    ggml_tensor* output,
    size_t count_per_peer
);

std::unique_ptr<Work> broadcast_async(
    ParallelContext& parallel,
    ggml_tensor* tensor,
    int root
);

} // namespace edgedit::ggml_comm

#endif // __ED_GGML_COMM_HPP__