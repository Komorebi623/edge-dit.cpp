#include "parallel/backends/nccl/nccl_process_group.hpp"

#include <chrono>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <thread>

namespace edgedit::parallel {
namespace {

void check_cuda(cudaError_t status, const char* expr) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(expr) + " failed: " + cudaGetErrorString(status));
    }
}

void check_nccl(ncclResult_t status, const char* expr) {
    if (status != ncclSuccess) {
        throw std::runtime_error(std::string(expr) + " failed: " + ncclGetErrorString(status));
    }
}

ncclDataType_t to_nccl_dtype(DataType type) {
    switch (type) {
        case DataType::kFloat32:
            return ncclFloat32;
        case DataType::kInt32:
            return ncclInt32;
    }
    throw std::invalid_argument("unsupported NCCL data type");
}

ncclRedOp_t to_nccl_reduce_op(ReduceOp op) {
    switch (op) {
        case ReduceOp::kSum:
            return ncclSum;
        case ReduceOp::kMax:
            return ncclMax;
        case ReduceOp::kMin:
            return ncclMin;
    }
    throw std::invalid_argument("unsupported NCCL reduce op");
}

void write_unique_id(const std::string& path, const ncclUniqueId& id) {
    const std::filesystem::path final_path(path);
    std::filesystem::create_directories(final_path.parent_path());
    const std::string tmp_path = path + ".tmp";
    {
        std::ofstream out(tmp_path, std::ios::binary | std::ios::trunc);
        if (!out) {
            throw std::runtime_error("failed to write NCCL unique id: " + tmp_path);
        }
        out.write(reinterpret_cast<const char*>(&id), sizeof(id));
    }
    std::filesystem::rename(tmp_path, final_path);
}

ncclUniqueId read_unique_id(const std::string& path) {
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(120);
    while (!std::filesystem::exists(path)) {
        if (std::chrono::steady_clock::now() > deadline) {
            throw std::runtime_error("timed out waiting for NCCL unique id: " + path);
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }

    ncclUniqueId id;
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        throw std::runtime_error("failed to read NCCL unique id: " + path);
    }
    in.read(reinterpret_cast<char*>(&id), sizeof(id));
    if (in.gcount() != static_cast<std::streamsize>(sizeof(id))) {
        throw std::runtime_error("invalid NCCL unique id size: " + path);
    }
    return id;
}

} // namespace

NcclProcessGroup::NcclProcessGroup(const ParallelConfig& config) : config_(config) {
    if (config_.world_size <= 0) {
        throw std::invalid_argument("NCCL world_size must be positive");
    }
    if (config_.store_path.empty() && config_.world_size > 1) {
        throw std::invalid_argument("NCCL backend requires store_path for unique id exchange");
    }

    check_cuda(cudaSetDevice(config_.device), "cudaSetDevice");
    check_cuda(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking), "cudaStreamCreateWithFlags");
    init_unique_id();
}

NcclProcessGroup::~NcclProcessGroup() {
    if (comm_ != nullptr) {
        ncclCommDestroy(comm_);
    }
    if (stream_ != nullptr) {
        cudaStreamDestroy(stream_);
    }
}

Backend NcclProcessGroup::backend() const {
    return Backend::kNccl;
}

int NcclProcessGroup::rank() const {
    return config_.rank;
}

int NcclProcessGroup::size() const {
    return config_.world_size;
}

bool NcclProcessGroup::enabled() const {
    return config_.world_size > 1;
}

void NcclProcessGroup::barrier() {
    int send = config_.rank;
    int recv = 0;
    Buffer in{&send, 1, DataType::kInt32, config_.device};
    Buffer out{&recv, 1, DataType::kInt32, config_.device};

    int* d_in  = nullptr;
    int* d_out = nullptr;
    check_cuda(cudaMalloc(&d_in, sizeof(int)), "cudaMalloc");
    check_cuda(cudaMalloc(&d_out, sizeof(int)), "cudaMalloc");
    check_cuda(cudaMemcpy(d_in, &send, sizeof(int), cudaMemcpyHostToDevice), "cudaMemcpy");
    in.data  = d_in;
    out.data = d_out;
    all_reduce(in, out, ReduceOp::kSum);
    check_cuda(cudaMemcpy(&recv, d_out, sizeof(int), cudaMemcpyDeviceToHost), "cudaMemcpy");
    check_cuda(cudaFree(d_in), "cudaFree");
    check_cuda(cudaFree(d_out), "cudaFree");
}

void NcclProcessGroup::all_reduce(const Buffer& input, const Buffer& output, ReduceOp op) {
    check_buffer(input);
    check_buffer(output);
    if (input.count != output.count || input.type != output.type) {
        throw std::invalid_argument("NCCL all_reduce input and output buffers must match");
    }
    check_nccl(ncclAllReduce(input.data,
                             output.data,
                             input.count,
                             to_nccl_dtype(input.type),
                             to_nccl_reduce_op(op),
                             comm_,
                             stream_),
               "ncclAllReduce");
    check_cuda(cudaStreamSynchronize(stream_), "cudaStreamSynchronize");
}

void NcclProcessGroup::all_gather(const Buffer& input, const Buffer& output) {
    check_buffer(input);
    check_buffer(output);
    if (input.type != output.type || output.count != input.count * static_cast<size_t>(config_.world_size)) {
        throw std::invalid_argument("NCCL all_gather output must hold world_size copies of input");
    }
    check_nccl(ncclAllGather(input.data, output.data, input.count, to_nccl_dtype(input.type), comm_, stream_),
               "ncclAllGather");
    check_cuda(cudaStreamSynchronize(stream_), "cudaStreamSynchronize");
}

void NcclProcessGroup::all_to_all(const Buffer& input, const Buffer& output, size_t count_per_peer) {
    check_buffer(input);
    check_buffer(output);
    const size_t total_count = count_per_peer * static_cast<size_t>(config_.world_size);
    if (input.type != output.type || input.count != total_count || output.count != total_count) {
        throw std::invalid_argument("NCCL all_to_all input and output must contain count_per_peer * world_size items");
    }

    const size_t item_size  = dtype_size(input.type);
    const size_t chunk_size = count_per_peer * item_size;
    const uint8_t* send     = reinterpret_cast<const uint8_t*>(input.data);
    uint8_t* recv           = reinterpret_cast<uint8_t*>(output.data);

    check_nccl(ncclGroupStart(), "ncclGroupStart");
    for (int peer = 0; peer < config_.world_size; ++peer) {
        check_nccl(ncclSend(send + static_cast<size_t>(peer) * chunk_size,
                            count_per_peer,
                            to_nccl_dtype(input.type),
                            peer,
                            comm_,
                            stream_),
                   "ncclSend");
        check_nccl(ncclRecv(recv + static_cast<size_t>(peer) * chunk_size,
                            count_per_peer,
                            to_nccl_dtype(output.type),
                            peer,
                            comm_,
                            stream_),
                   "ncclRecv");
    }
    check_nccl(ncclGroupEnd(), "ncclGroupEnd");
    check_cuda(cudaStreamSynchronize(stream_), "cudaStreamSynchronize");
}

void NcclProcessGroup::broadcast(const Buffer& buffer, int root) {
    check_buffer(buffer);
    if (root < 0 || root >= config_.world_size) {
        throw std::invalid_argument("NCCL broadcast root is out of range");
    }
    check_nccl(ncclBroadcast(buffer.data, buffer.data, buffer.count, to_nccl_dtype(buffer.type), root, comm_, stream_),
               "ncclBroadcast");
    check_cuda(cudaStreamSynchronize(stream_), "cudaStreamSynchronize");
}

void NcclProcessGroup::init_unique_id() {
    ncclUniqueId id;
    if (config_.world_size == 1) {
        check_nccl(ncclGetUniqueId(&id), "ncclGetUniqueId");
    } else if (config_.rank == 0) {
        check_nccl(ncclGetUniqueId(&id), "ncclGetUniqueId");
        write_unique_id(config_.store_path, id);
    } else {
        id = read_unique_id(config_.store_path);
    }
    check_nccl(ncclCommInitRank(&comm_, config_.world_size, id, config_.rank), "ncclCommInitRank");
}

void NcclProcessGroup::check_buffer(const Buffer& buffer) const {
    if (buffer.data == nullptr && buffer.count != 0) {
        throw std::invalid_argument("NCCL buffer data is null");
    }
}

} // namespace edgedit::parallel
