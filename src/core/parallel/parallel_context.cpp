#include "parallel/parallel_context.hpp"

#include <stdexcept>

namespace edgedit::parallel {

ParallelContext::ParallelContext(std::unique_ptr<ProcessGroup> group) : group_(std::move(group)) {
    if (!group_) {
        throw std::invalid_argument("parallel context requires a process group");
    }
}

bool ParallelContext::enabled() const {
    return group_->enabled();
}

int ParallelContext::rank() const {
    return group_->rank();
}

int ParallelContext::world_size() const {
    return group_->size();
}

ProcessGroup& ParallelContext::group() {
    return *group_;
}

const ProcessGroup& ParallelContext::group() const {
    return *group_;
}

std::unique_ptr<ParallelContext> create_parallel_context(const ParallelConfig& config) {
    return std::make_unique<ParallelContext>(create_process_group(config));
}

} // namespace edgedit::parallel
