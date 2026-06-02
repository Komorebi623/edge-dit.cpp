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

bool nearly_equal(float a, float b) {
    return std::fabs(a - b) < 1e-5f;
}

void check_vector(const std::vector<float>& got,
                  const std::vector<float>& expected,
                  const char* name) {
    if (got.size() != expected.size()) {
        throw std::runtime_error(std::string(name) + " size mismatch");
    }

    for (size_t i = 0; i < got.size(); ++i) {
        if (!nearly_equal(got[i], expected[i])) {
            throw std::runtime_error(
                std::string(name) +
                " mismatch at index " + std::to_string(i) +
                ": got " + std::to_string(got[i]) +
                ", expected " + std::to_string(expected[i])
            );
        }
    }
}

void run_host_tests(edgedit::parallel::ProcessGroup& group) {
    const int rank = group.rank();
    const int world_size = group.size();

    std::cout << "[rank " << rank << "] run host tests\n";

    // ------------------------------------------------------------
    // all_reduce sync
    // 每个 rank 输入 [rank + 1, (rank + 1) * 10]
    // sum 后应该是：
    // [1 + 2 + ... + world_size,
    //  10 * (1 + 2 + ... + world_size)]
    // ------------------------------------------------------------
    {
        std::vector<float> input = {
            static_cast<float>(rank + 1),
            static_cast<float>((rank + 1) * 10),
        };
        std::vector<float> output(input.size(), 0.0f);

        group.all_reduce(
            Buffer{input.data(), input.size(), DataType::kFloat32, -1},
            Buffer{output.data(), output.size(), DataType::kFloat32, -1},
            ReduceOp::kSum
        );

        const float sum_rank = static_cast<float>(world_size * (world_size + 1) / 2);
        std::vector<float> expected = {
            sum_rank,
            sum_rank * 10.0f,
        };

        check_vector(output, expected, "host all_reduce sync");
    }

    // ------------------------------------------------------------
    // all_reduce async
    // ------------------------------------------------------------
    {
        std::vector<float> input = {
            static_cast<float>(rank + 1),
            static_cast<float>((rank + 1) * 10),
        };
        std::vector<float> output(input.size(), 0.0f);

        auto work = group.all_reduce_async(
            Buffer{input.data(), input.size(), DataType::kFloat32, -1},
            Buffer{output.data(), output.size(), DataType::kFloat32, -1},
            ReduceOp::kSum
        );

        work->wait();

        const float sum_rank = static_cast<float>(world_size * (world_size + 1) / 2);
        std::vector<float> expected = {
            sum_rank,
            sum_rank * 10.0f,
        };

        check_vector(output, expected, "host all_reduce async");
    }

    // ------------------------------------------------------------
    // all_gather sync
    // rank r 输入 [r + 1, (r + 1) * 10]
    // gather 后 layout:
    // [rank0_data, rank1_data, ...]
    // ------------------------------------------------------------
    {
        std::vector<float> input = {
            static_cast<float>(rank + 1),
            static_cast<float>((rank + 1) * 10),
        };
        std::vector<float> output(input.size() * world_size, 0.0f);

        group.all_gather(
            Buffer{input.data(), input.size(), DataType::kFloat32, -1},
            Buffer{output.data(), output.size(), DataType::kFloat32, -1}
        );

        std::vector<float> expected;
        for (int r = 0; r < world_size; ++r) {
            expected.push_back(static_cast<float>(r + 1));
            expected.push_back(static_cast<float>((r + 1) * 10));
        }

        check_vector(output, expected, "host all_gather sync");
    }

    // ------------------------------------------------------------
    // all_gather async
    // ------------------------------------------------------------
    {
        std::vector<float> input = {
            static_cast<float>(rank + 1),
            static_cast<float>((rank + 1) * 10),
        };
        std::vector<float> output(input.size() * world_size, 0.0f);

        auto work = group.all_gather_async(
            Buffer{input.data(), input.size(), DataType::kFloat32, -1},
            Buffer{output.data(), output.size(), DataType::kFloat32, -1}
        );

        work->wait();

        std::vector<float> expected;
        for (int r = 0; r < world_size; ++r) {
            expected.push_back(static_cast<float>(r + 1));
            expected.push_back(static_cast<float>((r + 1) * 10));
        }

        check_vector(output, expected, "host all_gather async");
    }

    // ------------------------------------------------------------
    // all_to_all sync
    // 每个 rank 给每个 peer 发 1 个 float：
    // input[peer] = rank * 100 + peer
    //
    // rank k 收到：
    // [rank0 * 100 + k,
    //  rank1 * 100 + k,
    //  ...]
    // ------------------------------------------------------------
    {
        const size_t count_per_peer = 1;
        std::vector<float> input(world_size);
        std::vector<float> output(world_size, 0.0f);

        for (int peer = 0; peer < world_size; ++peer) {
            input[peer] = static_cast<float>(rank * 100 + peer);
        }

        group.all_to_all(
            Buffer{input.data(), input.size(), DataType::kFloat32, -1},
            Buffer{output.data(), output.size(), DataType::kFloat32, -1},
            count_per_peer
        );

        std::vector<float> expected;
        for (int src_rank = 0; src_rank < world_size; ++src_rank) {
            expected.push_back(static_cast<float>(src_rank * 100 + rank));
        }

        check_vector(output, expected, "host all_to_all sync");
    }

    // ------------------------------------------------------------
    // all_to_all async
    // ------------------------------------------------------------
    {
        const size_t count_per_peer = 1;
        std::vector<float> input(world_size);
        std::vector<float> output(world_size, 0.0f);

        for (int peer = 0; peer < world_size; ++peer) {
            input[peer] = static_cast<float>(rank * 100 + peer);
        }

        auto work = group.all_to_all_async(
            Buffer{input.data(), input.size(), DataType::kFloat32, -1},
            Buffer{output.data(), output.size(), DataType::kFloat32, -1},
            count_per_peer
        );

        work->wait();

        std::vector<float> expected;
        for (int src_rank = 0; src_rank < world_size; ++src_rank) {
            expected.push_back(static_cast<float>(src_rank * 100 + rank));
        }

        check_vector(output, expected, "host all_to_all async");
    }

    // ------------------------------------------------------------
    // broadcast sync
    // root = 0
    // ------------------------------------------------------------
    {
        std::vector<float> buffer = {
            rank == 0 ? 123.0f : 0.0f,
            rank == 0 ? 456.0f : 0.0f,
        };

        group.broadcast(
            Buffer{buffer.data(), buffer.size(), DataType::kFloat32, -1},
            0
        );

        std::vector<float> expected = {123.0f, 456.0f};
        check_vector(buffer, expected, "host broadcast sync");
    }

    // ------------------------------------------------------------
    // broadcast async
    // ------------------------------------------------------------
    {
        std::vector<float> buffer = {
            rank == 0 ? 123.0f : 0.0f,
            rank == 0 ? 456.0f : 0.0f,
        };

        auto work = group.broadcast_async(
            Buffer{buffer.data(), buffer.size(), DataType::kFloat32, -1},
            0
        );

        work->wait();

        std::vector<float> expected = {123.0f, 456.0f};
        check_vector(buffer, expected, "host broadcast async");
    }
}

#ifdef ED_ENABLE_NCCL

void check_cuda(cudaError_t status, const char* expr) {
    if (status != cudaSuccess) {
        throw std::runtime_error(
            std::string(expr) + " failed: " + cudaGetErrorString(status)
        );
    }
}

void run_nccl_tests(edgedit::parallel::ProcessGroup& group) {
    const int rank = group.rank();
    const int world_size = group.size();
    const int device = group.local_rank();

    std::cout << "[rank " << rank << "] run NCCL tests on device " << device << "\n";

    check_cuda(cudaSetDevice(device), "cudaSetDevice");

    // all_gather async test
    {
        std::vector<float> host_input = {
            static_cast<float>(rank + 1),
            static_cast<float>((rank + 1) * 10),
        };

        std::vector<float> host_output(host_input.size() * world_size, 0.0f);

        float* d_input = nullptr;
        float* d_output = nullptr;

        check_cuda(cudaMalloc(&d_input, host_input.size() * sizeof(float)), "cudaMalloc d_input");
        check_cuda(cudaMalloc(&d_output, host_output.size() * sizeof(float)), "cudaMalloc d_output");

        check_cuda(cudaMemcpy(d_input,
                              host_input.data(),
                              host_input.size() * sizeof(float),
                              cudaMemcpyHostToDevice),
                   "cudaMemcpy H2D");

        auto work = group.all_gather_async(
            Buffer{d_input, host_input.size(), DataType::kFloat32, device},
            Buffer{d_output, host_output.size(), DataType::kFloat32, device}
        );

        // 这里验证 async API 能返回 Work，真正同步在 wait
        if (work->is_completed()) {
            // 可能已经完成，不是错误
        }

        work->wait();

        check_cuda(cudaMemcpy(host_output.data(),
                              d_output,
                              host_output.size() * sizeof(float),
                              cudaMemcpyDeviceToHost),
                   "cudaMemcpy D2H");

        std::vector<float> expected;
        for (int r = 0; r < world_size; ++r) {
            expected.push_back(static_cast<float>(r + 1));
            expected.push_back(static_cast<float>((r + 1) * 10));
        }

        check_vector(host_output, expected, "NCCL all_gather async");

        cudaFree(d_input);
        cudaFree(d_output);
    }

    // all_reduce async test
    {
        std::vector<float> host_input = {
            static_cast<float>(rank + 1),
            static_cast<float>((rank + 1) * 10),
        };

        std::vector<float> host_output(host_input.size(), 0.0f);

        float* d_input = nullptr;
        float* d_output = nullptr;

        check_cuda(cudaMalloc(&d_input, host_input.size() * sizeof(float)), "cudaMalloc d_input");
        check_cuda(cudaMalloc(&d_output, host_output.size() * sizeof(float)), "cudaMalloc d_output");

        check_cuda(cudaMemcpy(d_input,
                              host_input.data(),
                              host_input.size() * sizeof(float),
                              cudaMemcpyHostToDevice),
                   "cudaMemcpy H2D");

        auto work = group.all_reduce_async(
            Buffer{d_input, host_input.size(), DataType::kFloat32, device},
            Buffer{d_output, host_output.size(), DataType::kFloat32, device},
            ReduceOp::kSum
        );

        work->wait();

        check_cuda(cudaMemcpy(host_output.data(),
                              d_output,
                              host_output.size() * sizeof(float),
                              cudaMemcpyDeviceToHost),
                   "cudaMemcpy D2H");

        const float sum_rank = static_cast<float>(world_size * (world_size + 1) / 2);
        std::vector<float> expected = {
            sum_rank,
            sum_rank * 10.0f,
        };

        check_vector(host_output, expected, "NCCL all_reduce async");

        cudaFree(d_input);
        cudaFree(d_output);
    }

    // all_to_all async test
    {
        const size_t count_per_peer = 1;
        std::vector<float> host_input(world_size);
        std::vector<float> host_output(world_size, 0.0f);

        for (int peer = 0; peer < world_size; ++peer) {
            host_input[peer] = static_cast<float>(rank * 100 + peer);
        }

        float* d_input = nullptr;
        float* d_output = nullptr;

        check_cuda(cudaMalloc(&d_input, host_input.size() * sizeof(float)), "cudaMalloc d_input");
        check_cuda(cudaMalloc(&d_output, host_output.size() * sizeof(float)), "cudaMalloc d_output");

        check_cuda(cudaMemcpy(d_input,
                              host_input.data(),
                              host_input.size() * sizeof(float),
                              cudaMemcpyHostToDevice),
                   "cudaMemcpy H2D");

        auto work = group.all_to_all_async(
            Buffer{d_input, host_input.size(), DataType::kFloat32, device},
            Buffer{d_output, host_output.size(), DataType::kFloat32, device},
            count_per_peer
        );

        work->wait();

        check_cuda(cudaMemcpy(host_output.data(),
                              d_output,
                              host_output.size() * sizeof(float),
                              cudaMemcpyDeviceToHost),
                   "cudaMemcpy D2H");

        std::vector<float> expected;
        for (int src_rank = 0; src_rank < world_size; ++src_rank) {
            expected.push_back(static_cast<float>(src_rank * 100 + rank));
        }

        check_vector(host_output, expected, "NCCL all_to_all async");

        cudaFree(d_input);
        cudaFree(d_output);
    }

    // broadcast async test
    {
        std::vector<float> host_buffer = {
            rank == 0 ? 123.0f : 0.0f,
            rank == 0 ? 456.0f : 0.0f,
        };

        float* d_buffer = nullptr;
        check_cuda(cudaMalloc(&d_buffer, host_buffer.size() * sizeof(float)), "cudaMalloc d_buffer");

        check_cuda(cudaMemcpy(d_buffer,
                              host_buffer.data(),
                              host_buffer.size() * sizeof(float),
                              cudaMemcpyHostToDevice),
                   "cudaMemcpy H2D");

        auto work = group.broadcast_async(
            Buffer{d_buffer, host_buffer.size(), DataType::kFloat32, device},
            0
        );

        work->wait();

        check_cuda(cudaMemcpy(host_buffer.data(),
                              d_buffer,
                              host_buffer.size() * sizeof(float),
                              cudaMemcpyDeviceToHost),
                   "cudaMemcpy D2H");

        std::vector<float> expected = {123.0f, 456.0f};
        check_vector(host_buffer, expected, "NCCL broadcast async");

        cudaFree(d_buffer);
    }
}

#endif

} // namespace

struct TestArgs {
    std::string backend = "cpu";
    int rank = -1;
    int world_size = -1;
    int local_rank = -1;
    int device = -1;
    std::string store_path;
};

int env_int_any(std::initializer_list<const char*> names, int fallback) {
    for (const char* name : names) {
        const char* value = std::getenv(name);
        if (value == nullptr || value[0] == '\0') {
            continue;
        }

        char* end = nullptr;
        long parsed = std::strtol(value, &end, 10);
        if (end != value && *end == '\0') {
            return static_cast<int>(parsed);
        }
    }
    return fallback;
}

TestArgs parse_test_args(int argc, char** argv) {
    TestArgs args;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];

        auto need_value = [&](const char* name) {
            if (i + 1 >= argc) {
                throw std::invalid_argument(std::string("missing value for ") + name);
            }
            return std::string(argv[++i]);
        };

        if (arg == "--backend") {
            args.backend = need_value("--backend");
        } else if (arg == "--rank") {
            args.rank = std::stoi(need_value("--rank"));
        } else if (arg == "--world-size") {
            args.world_size = std::stoi(need_value("--world-size"));
        } else if (arg == "--local-rank") {
            args.local_rank = std::stoi(need_value("--local-rank"));
        } else if (arg == "--device") {
            args.device = std::stoi(need_value("--device"));
        } else if (arg == "--store-path") {
            args.store_path = need_value("--store-path");
        } else if (arg == "--help" || arg == "-h") {
            std::cout
                << "Usage: parallel-collective-test [options]\n"
                << "Options:\n"
                << "  --backend cpu|nccl\n"
                << "  --rank N\n"
                << "  --world-size N\n"
                << "  --local-rank N\n"
                << "  --device N\n"
                << "  --store-path PATH\n";
            std::exit(0);
        } else {
            throw std::invalid_argument("unknown argument: " + arg);
        }
    }

    if (args.rank < 0) {
        args.rank = env_int_any({"RANK", "ED_RANK", "OMPI_COMM_WORLD_RANK", "PMI_RANK"}, 0);
    }
    if (args.world_size < 0) {
        args.world_size = env_int_any({"WORLD_SIZE", "ED_WORLD_SIZE", "OMPI_COMM_WORLD_SIZE", "PMI_SIZE"}, 1);
    }
    if (args.local_rank < 0) {
        args.local_rank = env_int_any({"LOCAL_RANK", "ED_LOCAL_RANK", "OMPI_COMM_WORLD_LOCAL_RANK", "MPI_LOCALRANKID"}, 0);
    }
    if (args.device < 0) {
        args.device = args.local_rank;
    }

    if (args.store_path.empty()) {
        const char* store_env = std::getenv("ED_STORE_PATH");
        if (store_env == nullptr || store_env[0] == '\0') {
            store_env = std::getenv("STORE_PATH");
        }
        if (store_env != nullptr && store_env[0] != '\0') {
            args.store_path = store_env;
        }
    }

    return args;
}

int main(int argc, char** argv) {
    try {
        TestArgs args = parse_test_args(argc, argv);

        ParallelConfig config;
        config.backend = edgedit::parallel::parse_backend(args.backend);
        config.rank = args.rank;
        config.world_size = args.world_size;
        config.local_rank = args.local_rank;
        config.device = args.device;
        config.store_path = args.store_path;

        std::cout
            << "[rank " << config.rank << "] config:"
            << " backend=" << args.backend
            << " world_size=" << config.world_size
            << " local_rank=" << config.local_rank
            << " device=" << config.device
            << " store_path=" << (config.store_path.empty() ? "<empty>" : config.store_path)
            << "\n";

        auto group = edgedit::parallel::create_process_group(config);

        if (group->backend() == Backend::kCpu) {
            run_host_tests(*group);
        }
#ifdef ED_ENABLE_NCCL
        else if (group->backend() == Backend::kNccl) {
            run_nccl_tests(*group);
        }
#endif
        else {
            throw std::runtime_error("unsupported test backend");
        }

        group->barrier();

        if (group->rank() == 0) {
            std::cout << "test_process_group passed\n";
        }

        return 0;
    } catch (const std::exception& e) {
        std::cerr << "test_process_group failed: " << e.what() << "\n";
        return 1;
    }
}