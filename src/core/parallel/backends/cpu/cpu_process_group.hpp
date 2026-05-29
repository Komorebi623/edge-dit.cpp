#ifndef __ED_PARALLEL_CPU_PROCESS_GROUP_HPP__
#define __ED_PARALLEL_CPU_PROCESS_GROUP_HPP__

#include "parallel/backends/cpu/file_store.hpp"
#include "parallel/process_group.hpp"

#include <atomic>
#include <memory>

namespace edgedit::parallel {

class CpuProcessGroup final : public ProcessGroup {
public:
    explicit CpuProcessGroup(const ParallelConfig& config);

    Backend backend() const override;
    int rank() const override;
    int size() const override;
    bool enabled() const override;

    void barrier() override;

    void all_reduce(const Buffer& input, const Buffer& output, ReduceOp op) override;
    void all_gather(const Buffer& input, const Buffer& output) override;
    void all_to_all(const Buffer& input, const Buffer& output, size_t count_per_peer) override;
    void broadcast(const Buffer& buffer, int root) override;

private:
    std::string next_key(const char* op);
    void validate_buffer(const Buffer& buffer) const;
    void reduce_payload(void* dst, const void* src, size_t count, DataType type, ReduceOp op, bool first) const;

    ParallelConfig config_;
    std::unique_ptr<FileStore> store_;
    std::atomic<uint64_t> op_index_{0};
};

} // namespace edgedit::parallel

#endif // __ED_PARALLEL_CPU_PROCESS_GROUP_HPP__
