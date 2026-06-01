#ifndef __ED_PARALLEL_CFG_PARALLEL_HPP__
#define __ED_PARALLEL_CFG_PARALLEL_HPP__

#include "parallel/parallel_context.hpp"
#include "utils/tensor.hpp"

#include <string>
#include <vector>

namespace edgedit::parallel {

bool cfg_parallel_available(const ParallelContext* context);
int cfg_parallel_rank(const ParallelContext* context);
int cfg_parallel_world_size(const ParallelContext* context);

bool cfg_all_gather(ParallelContext& context,
                    const sd::Tensor<float>& local,
                    std::vector<sd::Tensor<float>>* gathered,
                    std::string* error);

} // namespace edgedit::parallel

#endif // __ED_PARALLEL_CFG_PARALLEL_HPP__
