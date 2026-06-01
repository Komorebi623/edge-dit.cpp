#ifndef __ED_PARALLEL_NCCL_PROCESS_GROUP_HPP__
#define __ED_PARALLEL_NCCL_PROCESS_GROUP_HPP__

#include "parallel/process_group.hpp"

#include <atomic>
#include <string>

#include <cuda_runtime.h>
#include <nccl.h>

namespace edgedit::parallel {

class NcclProcessGroup final : public ProcessGroup {
public:
    explicit NcclProcessGroup(const ParallelConfig& config);
    ~NcclProcessGroup() override;

    NcclProcessGroup(const NcclProcessGroup&)            = delete;
    NcclProcessGroup& operator=(const NcclProcessGroup&) = delete;

    Backend backend() const override;
    int rank() const override;
    int size() const override;
    int local_rank() const override;
    bool enabled() const override;

    void barrier() override;

    void all_reduce(const Buffer& input, const Buffer& output, ReduceOp op) override;
    void all_gather(const Buffer& input, const Buffer& output) override;
    void all_to_all(const Buffer& input, const Buffer& output, size_t count_per_peer) override;
    void broadcast(const Buffer& buffer, int root) override;

private:
    void set_device() const;
    void init_unique_id();
    void check_buffer(const Buffer& buffer) const;

    ParallelConfig config_;
    ncclComm_t comm_     = nullptr;
    cudaStream_t stream_ = nullptr;
    std::atomic<uint64_t> op_index_{0};
};

} // namespace edgedit::parallel

#endif // __ED_PARALLEL_NCCL_PROCESS_GROUP_HPP__
