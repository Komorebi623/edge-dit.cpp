#include "parallel/process_group.hpp"

#include "parallel/backends/cpu/cpu_process_group.hpp"

#include <algorithm>
#include <cctype>
#include <stdexcept>

#ifdef ED_ENABLE_NCCL
#include "parallel/backends/nccl/nccl_process_group.hpp"
#endif

namespace edgedit::parallel {

const char* backend_name(Backend backend) {
    switch (backend) {
        case Backend::kNone:
            return "none";
        case Backend::kCpu:
            return "cpu";
        case Backend::kNccl:
            return "nccl";
    }
    return "unknown";
}

const char* dtype_name(DataType type) {
    switch (type) {
        case DataType::kFloat32:
            return "float32";
        case DataType::kInt32:
            return "int32";
    }
    return "unknown";
}

size_t dtype_size(DataType type) {
    switch (type) {
        case DataType::kFloat32:
            return sizeof(float);
        case DataType::kInt32:
            return sizeof(int32_t);
    }
    throw std::invalid_argument("unsupported parallel data type");
}

Backend parse_backend(const std::string& name) {
    std::string lower = name;
    std::transform(lower.begin(), lower.end(), lower.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });

    if (lower.empty() || lower == "none" || lower == "disabled") {
        return Backend::kNone;
    }
    if (lower == "cpu") {
        return Backend::kCpu;
    }
    if (lower == "nccl") {
        return Backend::kNccl;
    }
    throw std::invalid_argument("unknown parallel backend: " + name);
}

std::unique_ptr<ProcessGroup> create_process_group(const ParallelConfig& config) {
    if (config.world_size <= 0) {
        throw std::invalid_argument("parallel world_size must be positive");
    }
    if (config.rank < 0 || config.rank >= config.world_size) {
        throw std::invalid_argument("parallel rank must be in [0, world_size)");
    }

    switch (config.backend) {
        case Backend::kNone:
        case Backend::kCpu:
            return std::make_unique<CpuProcessGroup>(config);
        case Backend::kNccl:
#ifdef ED_ENABLE_NCCL
            return std::make_unique<NcclProcessGroup>(config);
#else
            throw std::runtime_error("NCCL backend requested, but edge-dit was built without ED_ENABLE_NCCL");
#endif
    }
    throw std::invalid_argument("unsupported parallel backend");
}

} // namespace edgedit::parallel
