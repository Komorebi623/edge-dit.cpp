#ifndef __ED_PARALLEL_CONTEXT_HPP__
#define __ED_PARALLEL_CONTEXT_HPP__

#include "parallel/process_group.hpp"

#include <memory>

namespace edgedit::parallel {

class ParallelContext {
public:
    explicit ParallelContext(std::unique_ptr<ProcessGroup> group);

    bool enabled() const;
    int rank() const;
    int world_size() const;

    ProcessGroup& group();
    const ProcessGroup& group() const;

private:
    std::unique_ptr<ProcessGroup> group_;
};

std::unique_ptr<ParallelContext> create_parallel_context(const ParallelConfig& config);

} // namespace edgedit::parallel

#endif // __ED_PARALLEL_CONTEXT_HPP__
