#include "parallel/backends/cpu/cpu_process_group.hpp"

#include <algorithm>
#include <cstring>
#include <stdexcept>

namespace edgedit::parallel {
namespace {

template <typename T>
void reduce_typed(T* dst, const T* src, size_t count, ReduceOp op, bool first) {
    if (first) {
        std::memcpy(dst, src, count * sizeof(T));
        return;
    }

    for (size_t i = 0; i < count; ++i) {
        switch (op) {
            case ReduceOp::kSum:
                dst[i] += src[i];
                break;
            case ReduceOp::kMax:
                dst[i] = std::max(dst[i], src[i]);
                break;
            case ReduceOp::kMin:
                dst[i] = std::min(dst[i], src[i]);
                break;
        }
    }
}

} // namespace

CpuProcessGroup::CpuProcessGroup(const ParallelConfig& config) : config_(config) {
    if (config_.backend == Backend::kNone && config_.world_size != 1) {
        throw std::invalid_argument("parallel backend 'none' only supports world_size=1");
    }

    if (config_.world_size > 1 || config_.backend == Backend::kCpu) {
        if (config_.store_path.empty()) {
            throw std::invalid_argument("cpu parallel backend requires a store path");
        }
        store_ = std::make_unique<FileStore>(config_.store_path, config_.rank, config_.world_size);
    }
}

Backend CpuProcessGroup::backend() const {
    return config_.backend;
}

int CpuProcessGroup::rank() const {
    return config_.rank;
}

int CpuProcessGroup::size() const {
    return config_.world_size;
}

bool CpuProcessGroup::enabled() const {
    return config_.backend != Backend::kNone && config_.world_size > 1;
}

void CpuProcessGroup::barrier() {
    if (!enabled()) {
        return;
    }
    store_->barrier(next_key("barrier"));
}

void CpuProcessGroup::all_reduce(const Buffer& input, const Buffer& output, ReduceOp op) {
    validate_buffer(input);
    validate_buffer(output);
    if (input.count != output.count || input.type != output.type) {
        throw std::invalid_argument("all_reduce input and output buffers must have matching shape and type");
    }

    if (!enabled()) {
        std::memcpy(output.data, input.data, input.count * dtype_size(input.type));
        return;
    }

    const std::string key  = next_key("all_reduce");
    const size_t nbytes    = input.count * dtype_size(input.type);
    const auto rank_suffix = std::to_string(config_.rank);
    store_->write_bytes(key + "_rank_" + rank_suffix, input.data, nbytes);

    for (int r = 0; r < config_.world_size; ++r) {
        auto payload = store_->read_bytes(key + "_rank_" + std::to_string(r), nbytes);
        reduce_payload(output.data, payload.data(), output.count, output.type, op, r == 0);
    }
    store_->barrier(key + "_done");
}

void CpuProcessGroup::all_gather(const Buffer& input, const Buffer& output) {
    validate_buffer(input);
    validate_buffer(output);
    if (input.type != output.type || output.count != input.count * static_cast<size_t>(config_.world_size)) {
        throw std::invalid_argument("all_gather output must hold world_size copies of the input");
    }

    if (!enabled()) {
        std::memcpy(output.data, input.data, input.count * dtype_size(input.type));
        return;
    }

    const std::string key = next_key("all_gather");
    const size_t nbytes   = input.count * dtype_size(input.type);
    store_->write_bytes(key + "_rank_" + std::to_string(config_.rank), input.data, nbytes);

    uint8_t* dst = reinterpret_cast<uint8_t*>(output.data);
    for (int r = 0; r < config_.world_size; ++r) {
        auto payload = store_->read_bytes(key + "_rank_" + std::to_string(r), nbytes);
        std::memcpy(dst + static_cast<size_t>(r) * nbytes, payload.data(), nbytes);
    }
    store_->barrier(key + "_done");
}

void CpuProcessGroup::all_to_all(const Buffer& input, const Buffer& output, size_t count_per_peer) {
    validate_buffer(input);
    validate_buffer(output);
    const size_t total_count = count_per_peer * static_cast<size_t>(config_.world_size);
    if (input.type != output.type || input.count != total_count || output.count != total_count) {
        throw std::invalid_argument("all_to_all input and output must both contain count_per_peer * world_size items");
    }

    if (!enabled()) {
        std::memcpy(output.data, input.data, input.count * dtype_size(input.type));
        return;
    }

    const std::string key = next_key("all_to_all");
    const size_t nbytes   = input.count * dtype_size(input.type);
    store_->write_bytes(key + "_rank_" + std::to_string(config_.rank), input.data, nbytes);

    const size_t item_size  = dtype_size(input.type);
    const size_t chunk_size = count_per_peer * item_size;
    uint8_t* dst            = reinterpret_cast<uint8_t*>(output.data);
    for (int src_rank = 0; src_rank < config_.world_size; ++src_rank) {
        auto payload = store_->read_bytes(key + "_rank_" + std::to_string(src_rank), nbytes);
        const uint8_t* src =
            payload.data() + static_cast<size_t>(config_.rank) * count_per_peer * item_size;
        std::memcpy(dst + static_cast<size_t>(src_rank) * chunk_size, src, chunk_size);
    }
    store_->barrier(key + "_done");
}

void CpuProcessGroup::broadcast(const Buffer& buffer, int root) {
    validate_buffer(buffer);
    if (root < 0 || root >= config_.world_size) {
        throw std::invalid_argument("broadcast root is out of range");
    }

    if (!enabled()) {
        return;
    }

    const std::string key = next_key("broadcast");
    const size_t nbytes   = buffer.count * dtype_size(buffer.type);
    if (config_.rank == root) {
        store_->write_bytes(key + "_root", buffer.data, nbytes);
    } else {
        auto payload = store_->read_bytes(key + "_root", nbytes);
        std::memcpy(buffer.data, payload.data(), nbytes);
    }
    store_->barrier(key + "_done");
}

std::string CpuProcessGroup::next_key(const char* op) {
    const uint64_t index = op_index_.fetch_add(1);
    return std::string(op) + "_" + std::to_string(index);
}

void CpuProcessGroup::validate_buffer(const Buffer& buffer) const {
    if (buffer.data == nullptr && buffer.count != 0) {
        throw std::invalid_argument("parallel buffer data is null");
    }
}

void CpuProcessGroup::reduce_payload(void* dst,
                                     const void* src,
                                     size_t count,
                                     DataType type,
                                     ReduceOp op,
                                     bool first) const {
    switch (type) {
        case DataType::kFloat32:
            reduce_typed(reinterpret_cast<float*>(dst), reinterpret_cast<const float*>(src), count, op, first);
            return;
        case DataType::kInt32:
            reduce_typed(reinterpret_cast<int32_t*>(dst), reinterpret_cast<const int32_t*>(src), count, op, first);
            return;
    }
    throw std::invalid_argument("unsupported all_reduce data type");
}

} // namespace edgedit::parallel
