#include "dit_models/models/minimax_h3_cuda_options.hpp"
#include <iostream>
#include <map>
#include <stdexcept>
using namespace MiniMaxH3;
static void check(bool ok, const char* what) { if (!ok) throw std::runtime_error(what); }
int main() {
    try {
        std::map<std::string, std::string> env;
        auto lookup = [&](const char* name) { auto it = env.find(name); return it == env.end() ? std::string{} : it->second; };
        CudaOptimizationOptions opts;
        CudaOptimizationCapabilities caps;
        std::string error;
        check(parse_cuda_optimization_options(lookup, &opts, &error) && !opts.any_enabled(), "defaults must be off");
        check(validate_cuda_optimization_options(opts, false, caps, &error), "CPU baseline must be unchanged");
        const char* flags[] = {"ED_H3_FUSED_OPS", "ED_H3_CACHE_DIT", "ED_H3_ADALN_CACHE", "ED_H3_SAGE_ATTN", "ED_H3_SPARSE_ATTN", "ED_H3_VDN", "ED_H3_VSA", "ED_H3_TURBO"};
        for (auto flag : flags) {
            env = {{flag, "1"}};
            check(parse_cuda_optimization_options(lookup, &opts, &error) && opts.any_enabled(), "flag parse failed");
            check(!validate_cuda_optimization_options(opts, false, caps, &error), "optimization escaped CUDA gate");
            check(!validate_cuda_optimization_options(opts, true, caps, &error) && error.find(flag) != std::string::npos, "unsupported silently enabled");
            env[flag] = "yes";
            check(!parse_cuda_optimization_options(lookup, &opts, &error), "malformed flag accepted");
        }
        env = {{"ED_H3_FUSED_OPS", "1"}};
        caps.fused_ops = true;
        check(parse_cuda_optimization_options(lookup, &opts, &error) && validate_cuda_optimization_options(opts, true, caps, &error), "supported fusion rejected");
        env = {{"ED_H3_CACHE_DIT", "1"}, {"ED_H3_CACHE_THRESHOLD", "nan"}};
        check(!parse_cuda_optimization_options(lookup, &opts, &error), "NaN accepted");
        env["ED_H3_CACHE_THRESHOLD"] = "0.04";
        env["ED_H3_CACHE_FRONT_BLOCKS"] = "2x";
        check(!parse_cuda_optimization_options(lookup, &opts, &error), "non-integer accepted");
        env["ED_H3_CACHE_FRONT_BLOCKS"] = "2";
        env["ED_H3_CACHE_START_PERCENT"] = "0.9";
        env["ED_H3_CACHE_END_PERCENT"] = "0.1";
        check(!parse_cuda_optimization_options(lookup, &opts, &error), "reversed cache window accepted");
        env = {{"ED_H3_CACHE_THRESHOLD", "malformed-but-inactive"}};
        check(parse_cuda_optimization_options(lookup, &opts, &error) && !opts.any_enabled(), "inactive tuning changes baseline");
        env = {{"ED_SAGE_ATTN", "1"}, {"ED_OTHER_MODEL_FLAG", "1"}};
        check(parse_cuda_optimization_options(lookup, &opts, &error) && !opts.any_enabled(), "unrelated env changes H3 policy");
        check(!parse_cuda_optimization_options(lookup, nullptr, &error), "missing output accepted");
        std::cout << "H3 CUDA policy tests passed\n";
    } catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
