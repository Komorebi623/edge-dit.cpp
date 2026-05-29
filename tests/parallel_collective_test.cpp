#include "parallel/process_group.hpp"

#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef ED_ENABLE_NCCL
#include <cuda_runtime.h>
#endif

using edgedit::parallel::Backend;
using edgedit::parallel::Buffer;
using edgedit::parallel::DataType;
using edgedit::parallel::ParallelConfig;
using edgedit::parallel::ReduceOp;

namespace {

struct Args {
    std::string backend = "cpu";
    int rank            = 0;
    int world_size      = 1;
    int local_rank      = 0;
    int device          = 0;
    std::string store_path;
};

int env_int(const char* name, int fallback) {
    const char* value = std::getenv(name);
    if (value == nullptr || value[0] == '\0') {
        return fallback;
    }
    char* end = nullptr;
    long parsed = std::strtol(value, &end, 10);
    if (end == value || *end != '\0') {
        return fallback;
    }
    return static_cast<int>(parsed);
}

int infer_local_rank(int fallback) {
    fallback = env_int("LOCAL_RANK", fallback);
    fallback = env_int("OMPI_COMM_WORLD_LOCAL_RANK", fallback);
    fallback = env_int("MV2_COMM_WORLD_LOCAL_RANK", fallback);
    fallback = env_int("SLURM_LOCALID", fallback);
    fallback = env_int("PMI_LOCAL_RANK", fallback);
    return fallback;
}

int parse_int(const char* value, const char* name) {
    char* end = nullptr;
    long v    = std::strtol(value, &end, 10);
    if (end == value || *end != '\0') {
        throw std::invalid_argument(std::string("invalid integer for ") + name + ": " + value);
    }
    return static_cast<int>(v);
}

Args parse_args(int argc, char** argv) {
    Args args;
    for (int i = 1; i < argc; ++i) {
        auto require_value = [&](const char* name) -> const char* {
            if (i + 1 >= argc) {
                throw std::invalid_argument(std::string("missing value for ") + name);
            }
            return argv[++i];
        };

        if (std::strcmp(argv[i], "--backend") == 0) {
            args.backend = require_value(argv[i]);
        } else if (std::strcmp(argv[i], "--rank") == 0) {
            args.rank = parse_int(require_value(argv[i]), argv[i]);
        } else if (std::strcmp(argv[i], "--world-size") == 0) {
            args.world_size = parse_int(require_value(argv[i]), argv[i]);
        } else if (std::strcmp(argv[i], "--local-rank") == 0) {
            args.local_rank = parse_int(require_value(argv[i]), argv[i]);
        } else if (std::strcmp(argv[i], "--device") == 0) {
            args.device = parse_int(require_value(argv[i]), argv[i]);
        } else if (std::strcmp(argv[i], "--store-path") == 0) {
            args.store_path = require_value(argv[i]);
        } else {
            throw std::invalid_argument(std::string("unknown argument: ") + argv[i]);
        }
    }
    return args;
}

bool nearly_equal(float a, float b) {
    return std::fabs(a - b) < 1e-5f;
}

void check_vector(const std::vector<float>& got, const std::vector<float>& expected, const char* name) {
    if (got.size() != expected.size()) {
        throw std::runtime_error(std::string(name) + " size mismatch");
    }
    for (size_t i = 0; i < got.size(); ++i) {
        if (!nearly_equal(got[i], expected[i])) {
            throw std::runtime_error(std::string(name) + " mismatch at index " + std::to_string(i) +
                                     ": got " + std::to_string(got[i]) +
                                     ", expected " + std::to_string(expected[i]));
        }
    }
}

void run_host_tests(edgedit::parallel::ProcessGroup& group) {
    const int rank       = group.rank();
    const int world_size = group.size();

    std::vector<float> input = {static_cast<float>(rank + 1), static_cast<float>((rank + 1) * 10)};
    std::vector<float> reduced(input.size(), 0.0f);
    group.all_reduce(Buffer{input.data(), input.size(), DataType::kFloat32, -1},
                     Buffer{reduced.data(), reduced.size(), DataType::kFloat32, -1},
                     ReduceOp::kSum);

    const float sum_ranks = static_cast<float>(world_size * (world_size + 1) / 2);
    check_vector(reduced, {sum_ranks, sum_ranks * 10.0f}, "all_reduce");

    std::vector<float> gathered(static_cast<size_t>(world_size) * input.size(), 0.0f);
    group.all_gather(Buffer{input.data(), input.size(), DataType::kFloat32, -1},
                     Buffer{gathered.data(), gathered.size(), DataType::kFloat32, -1});
    std::vector<float> expected_gathered;
    for (int r = 0; r < world_size; ++r) {
        expected_gathered.push_back(static_cast<float>(r + 1));
        expected_gathered.push_back(static_cast<float>((r + 1) * 10));
    }
    check_vector(gathered, expected_gathered, "all_gather");

    std::vector<float> a2a(static_cast<size_t>(world_size), 0.0f);
    for (int peer = 0; peer < world_size; ++peer) {
        a2a[peer] = static_cast<float>(rank * 100 + peer);
    }
    std::vector<float> a2a_out(a2a.size(), 0.0f);
    group.all_to_all(Buffer{a2a.data(), a2a.size(), DataType::kFloat32, -1},
                     Buffer{a2a_out.data(), a2a_out.size(), DataType::kFloat32, -1},
                     1);
    std::vector<float> expected_a2a;
    for (int src = 0; src < world_size; ++src) {
        expected_a2a.push_back(static_cast<float>(src * 100 + rank));
    }
    check_vector(a2a_out, expected_a2a, "all_to_all");

    std::vector<float> broadcast = {rank == 0 ? 123.0f : -1.0f};
    group.broadcast(Buffer{broadcast.data(), broadcast.size(), DataType::kFloat32, -1}, 0);
    check_vector(broadcast, {123.0f}, "broadcast");

    group.barrier();
}

#ifdef ED_ENABLE_NCCL
void check_cuda(cudaError_t status, const char* expr) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(expr) + " failed: " + cudaGetErrorString(status));
    }
}

std::vector<float> copy_from_device(float* ptr, size_t count) {
    std::vector<float> host(count);
    check_cuda(cudaMemcpy(host.data(), ptr, count * sizeof(float), cudaMemcpyDeviceToHost), "cudaMemcpy");
    return host;
}

void run_cuda_tests(edgedit::parallel::ProcessGroup& group, int device) {
    check_cuda(cudaSetDevice(device), "cudaSetDevice");
    const int rank       = group.rank();
    const int world_size = group.size();

    std::vector<float> h_input = {static_cast<float>(rank + 1), static_cast<float>((rank + 1) * 10)};
    float* d_input             = nullptr;
    float* d_output            = nullptr;
    float* d_gather            = nullptr;
    float* d_a2a               = nullptr;
    float* d_a2a_out           = nullptr;
    float* d_broadcast         = nullptr;

    check_cuda(cudaMalloc(&d_input, h_input.size() * sizeof(float)), "cudaMalloc");
    check_cuda(cudaMalloc(&d_output, h_input.size() * sizeof(float)), "cudaMalloc");
    check_cuda(cudaMalloc(&d_gather, h_input.size() * static_cast<size_t>(world_size) * sizeof(float)), "cudaMalloc");
    check_cuda(cudaMalloc(&d_a2a, static_cast<size_t>(world_size) * sizeof(float)), "cudaMalloc");
    check_cuda(cudaMalloc(&d_a2a_out, static_cast<size_t>(world_size) * sizeof(float)), "cudaMalloc");
    check_cuda(cudaMalloc(&d_broadcast, sizeof(float)), "cudaMalloc");

    check_cuda(cudaMemcpy(d_input, h_input.data(), h_input.size() * sizeof(float), cudaMemcpyHostToDevice),
               "cudaMemcpy");
    group.all_reduce(Buffer{d_input, h_input.size(), DataType::kFloat32, device},
                     Buffer{d_output, h_input.size(), DataType::kFloat32, device},
                     ReduceOp::kSum);
    const float sum_ranks = static_cast<float>(world_size * (world_size + 1) / 2);
    check_vector(copy_from_device(d_output, h_input.size()), {sum_ranks, sum_ranks * 10.0f}, "cuda all_reduce");

    group.all_gather(Buffer{d_input, h_input.size(), DataType::kFloat32, device},
                     Buffer{d_gather, h_input.size() * static_cast<size_t>(world_size), DataType::kFloat32, device});
    std::vector<float> expected_gathered;
    for (int r = 0; r < world_size; ++r) {
        expected_gathered.push_back(static_cast<float>(r + 1));
        expected_gathered.push_back(static_cast<float>((r + 1) * 10));
    }
    check_vector(copy_from_device(d_gather, expected_gathered.size()), expected_gathered, "cuda all_gather");

    std::vector<float> h_a2a(static_cast<size_t>(world_size));
    for (int peer = 0; peer < world_size; ++peer) {
        h_a2a[peer] = static_cast<float>(rank * 100 + peer);
    }
    check_cuda(cudaMemcpy(d_a2a, h_a2a.data(), h_a2a.size() * sizeof(float), cudaMemcpyHostToDevice), "cudaMemcpy");
    group.all_to_all(Buffer{d_a2a, h_a2a.size(), DataType::kFloat32, device},
                     Buffer{d_a2a_out, h_a2a.size(), DataType::kFloat32, device},
                     1);
    std::vector<float> expected_a2a;
    for (int src = 0; src < world_size; ++src) {
        expected_a2a.push_back(static_cast<float>(src * 100 + rank));
    }
    check_vector(copy_from_device(d_a2a_out, expected_a2a.size()), expected_a2a, "cuda all_to_all");

    float h_broadcast = rank == 0 ? 123.0f : -1.0f;
    check_cuda(cudaMemcpy(d_broadcast, &h_broadcast, sizeof(float), cudaMemcpyHostToDevice), "cudaMemcpy");
    group.broadcast(Buffer{d_broadcast, 1, DataType::kFloat32, device}, 0);
    check_vector(copy_from_device(d_broadcast, 1), {123.0f}, "cuda broadcast");
    group.barrier();

    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_gather);
    cudaFree(d_a2a);
    cudaFree(d_a2a_out);
    cudaFree(d_broadcast);
}
#endif

} // namespace

int main(int argc, char** argv) {
    try {
        Args args = parse_args(argc, argv);
        ParallelConfig config;
        config.backend    = edgedit::parallel::parse_backend(args.backend);
        config.rank       = args.rank;
        config.world_size = args.world_size;
        config.local_rank = args.local_rank;
        config.device     = args.device;
        config.store_path = args.store_path;

        auto group = edgedit::parallel::create_process_group(config);
        if (config.backend == Backend::kNccl) {
#ifdef ED_ENABLE_NCCL
            run_cuda_tests(*group, infer_local_rank(args.device));
#else
            throw std::runtime_error("binary was built without NCCL support");
#endif
        } else {
            run_host_tests(*group);
        }

        std::cout << "rank " << group->rank() << "/" << group->size() << " backend=" << args.backend << " ok\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "parallel_collective_test failed: " << e.what() << "\n";
        return 1;
    }
}
