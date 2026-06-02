#include "backend/ggml/parallel/ggml_comm.hpp"
#include "parallel/process_group.hpp"

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <cmath>
#include <cstdlib>
#include <cstring>
#include <initializer_list>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

using edgedit::parallel::Backend;
using edgedit::parallel::ParallelConfig;
using edgedit::parallel::ReduceOp;

namespace {

struct Args {
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

std::string env_str_any(std::initializer_list<const char*> names, const std::string& fallback = "") {
    for (const char* name : names) {
        const char* value = std::getenv(name);
        if (value != nullptr && value[0] != '\0') {
            return std::string(value);
        }
    }
    return fallback;
}

Args parse_args(int argc, char** argv) {
    Args args;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];

        auto need_value = [&](const char* name) -> std::string {
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
                << "Usage: ggml-comm-test [options]\n"
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
        args.store_path = env_str_any({"ED_STORE_PATH", "STORE_PATH"}, "");
    }

    return args;
}

void check_close(const std::vector<float>& got,
                 const std::vector<float>& expected,
                 const char* name) {
    if (got.size() != expected.size()) {
        throw std::runtime_error(std::string(name) + " size mismatch: got=" +
                                 std::to_string(got.size()) +
                                 " expected=" +
                                 std::to_string(expected.size()));
    }

    for (size_t i = 0; i < got.size(); ++i) {
        if (std::fabs(got[i] - expected[i]) > 1e-5f) {
            throw std::runtime_error(
                std::string(name) +
                " mismatch at index " + std::to_string(i) +
                ", got=" + std::to_string(got[i]) +
                ", expected=" + std::to_string(expected[i])
            );
        }
    }
}

ggml_tensor* new_1d_f32(ggml_context* ctx, int64_t n, const char* name) {
    ggml_tensor* t = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n);
    ggml_set_name(t, name);
    return t;
}

void tensor_set_f32(ggml_tensor* tensor, const std::vector<float>& values) {
    const size_t nbytes = values.size() * sizeof(float);
    if (static_cast<size_t>(ggml_nelements(tensor)) != values.size()) {
        throw std::runtime_error("tensor_set_f32 size mismatch");
    }
    ggml_backend_tensor_set(tensor, values.data(), 0, nbytes);
}

std::vector<float> tensor_get_f32(ggml_tensor* tensor) {
    std::vector<float> values(static_cast<size_t>(ggml_nelements(tensor)));
    ggml_backend_tensor_get(tensor, values.data(), 0, values.size() * sizeof(float));
    return values;
}

ggml_backend_t init_ggml_backend_for_test(const Args& args) {
    if (args.backend == "cpu") {
        ggml_backend_t backend = ggml_backend_cpu_init();
        if (backend == nullptr) {
            throw std::runtime_error("ggml_backend_cpu_init failed");
        }
        return backend;
    }

    if (args.backend == "nccl") {
        ggml_backend_load_all();

        // CUDA_VISIBLE_DEVICES=0,1 后，ggml 里通常对应 CUDA0 / CUDA1。
        // local_rank=0 -> CUDA0, local_rank=1 -> CUDA1。
        const std::string cuda_name = "CUDA" + std::to_string(args.local_rank);
        ggml_backend_t backend = ggml_backend_init_by_name(cuda_name.c_str(), nullptr);

        if (backend == nullptr) {
            std::ostringstream oss;
            oss << "failed to init ggml CUDA backend by name: " << cuda_name
                << ". Make sure the project is configured with -DED_GGML_CUDA=ON.";
            throw std::runtime_error(oss.str());
        }

        return backend;
    }

    throw std::runtime_error("unsupported backend for ggml_comm_test: " + args.backend);
}

ggml_context* new_test_context() {
    ggml_init_params params;
    params.mem_size = 16 * 1024 * 1024;
    params.mem_buffer = nullptr;
    params.no_alloc = true;

    ggml_context* ctx = ggml_init(params);
    if (ctx == nullptr) {
        throw std::runtime_error("ggml_init failed");
    }
    return ctx;
}

void run_all_reduce_test(edgedit::parallel::ProcessGroup& group,
                         ggml_backend_t backend) {
    const int rank = group.rank();
    const int world_size = group.size();

    ggml_context* ctx = new_test_context();

    ggml_tensor* input = new_1d_f32(ctx, 2, "all_reduce_input");
    ggml_tensor* output = new_1d_f32(ctx, 2, "all_reduce_output");

    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (buffer == nullptr) {
        ggml_free(ctx);
        throw std::runtime_error("ggml_backend_alloc_ctx_tensors failed in all_reduce");
    }

    tensor_set_f32(input, {
        static_cast<float>(rank + 1),
        static_cast<float>((rank + 1) * 10),
    });

    std::string error;
    if (!edgedit::ggml_comm::all_reduce(group, input, output, ReduceOp::kSum, &error)) {
        ggml_backend_buffer_free(buffer);
        ggml_free(ctx);
        throw std::runtime_error("ggml_comm all_reduce failed: " + error);
    }

    std::vector<float> got = tensor_get_f32(output);
    const float sum_rank = static_cast<float>(world_size * (world_size + 1) / 2);
    std::vector<float> expected = {sum_rank, sum_rank * 10.0f};
    check_close(got, expected, "ggml_comm all_reduce");

    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
}

void run_all_gather_test(edgedit::parallel::ProcessGroup& group,
                         ggml_backend_t backend) {
    const int rank = group.rank();
    const int world_size = group.size();

    ggml_context* ctx = new_test_context();

    ggml_tensor* input = new_1d_f32(ctx, 2, "all_gather_input");
    ggml_tensor* output = new_1d_f32(ctx, 2 * world_size, "all_gather_output");

    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (buffer == nullptr) {
        ggml_free(ctx);
        throw std::runtime_error("ggml_backend_alloc_ctx_tensors failed in all_gather");
    }

    tensor_set_f32(input, {
        static_cast<float>(rank + 1),
        static_cast<float>((rank + 1) * 10),
    });

    std::string error;
    if (!edgedit::ggml_comm::all_gather(group, input, output, &error)) {
        ggml_backend_buffer_free(buffer);
        ggml_free(ctx);
        throw std::runtime_error("ggml_comm all_gather failed: " + error);
    }

    std::vector<float> got = tensor_get_f32(output);

    std::vector<float> expected;
    for (int r = 0; r < world_size; ++r) {
        expected.push_back(static_cast<float>(r + 1));
        expected.push_back(static_cast<float>((r + 1) * 10));
    }

    check_close(got, expected, "ggml_comm all_gather");

    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
}

void run_all_to_all_test(edgedit::parallel::ProcessGroup& group,
                         ggml_backend_t backend) {
    const int rank = group.rank();
    const int world_size = group.size();
    const size_t count_per_peer = 1;

    ggml_context* ctx = new_test_context();

    ggml_tensor* input = new_1d_f32(ctx, world_size, "all_to_all_input");
    ggml_tensor* output = new_1d_f32(ctx, world_size, "all_to_all_output");

    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (buffer == nullptr) {
        ggml_free(ctx);
        throw std::runtime_error("ggml_backend_alloc_ctx_tensors failed in all_to_all");
    }

    std::vector<float> input_values(static_cast<size_t>(world_size));
    for (int peer = 0; peer < world_size; ++peer) {
        input_values[static_cast<size_t>(peer)] =
            static_cast<float>(rank * 100 + peer);
    }
    tensor_set_f32(input, input_values);

    std::string error;
    if (!edgedit::ggml_comm::all_to_all(group, input, output, count_per_peer, &error)) {
        ggml_backend_buffer_free(buffer);
        ggml_free(ctx);
        throw std::runtime_error("ggml_comm all_to_all failed: " + error);
    }

    std::vector<float> got = tensor_get_f32(output);

    std::vector<float> expected;
    for (int src_rank = 0; src_rank < world_size; ++src_rank) {
        expected.push_back(static_cast<float>(src_rank * 100 + rank));
    }

    check_close(got, expected, "ggml_comm all_to_all");

    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
}

void run_broadcast_test(edgedit::parallel::ProcessGroup& group,
                        ggml_backend_t backend) {
    const int rank = group.rank();

    ggml_context* ctx = new_test_context();

    ggml_tensor* tensor = new_1d_f32(ctx, 2, "broadcast_tensor");

    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (buffer == nullptr) {
        ggml_free(ctx);
        throw std::runtime_error("ggml_backend_alloc_ctx_tensors failed in broadcast");
    }

    if (rank == 0) {
        tensor_set_f32(tensor, {123.0f, 456.0f});
    } else {
        tensor_set_f32(tensor, {0.0f, 0.0f});
    }

    std::string error;
    if (!edgedit::ggml_comm::broadcast(group, tensor, 0, &error)) {
        ggml_backend_buffer_free(buffer);
        ggml_free(ctx);
        throw std::runtime_error("ggml_comm broadcast failed: " + error);
    }

    std::vector<float> got = tensor_get_f32(tensor);
    std::vector<float> expected = {123.0f, 456.0f};
    check_close(got, expected, "ggml_comm broadcast");

    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
}

void run_async_all_gather_test(edgedit::parallel::ProcessGroup& group,
                               ggml_backend_t backend) {
    const int rank = group.rank();
    const int world_size = group.size();

    ggml_context* ctx = new_test_context();

    ggml_tensor* input = new_1d_f32(ctx, 2, "async_all_gather_input");
    ggml_tensor* output = new_1d_f32(ctx, 2 * world_size, "async_all_gather_output");

    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (buffer == nullptr) {
        ggml_free(ctx);
        throw std::runtime_error("ggml_backend_alloc_ctx_tensors failed in async_all_gather");
    }

    tensor_set_f32(input, {
        static_cast<float>(rank + 1),
        static_cast<float>((rank + 1) * 10),
    });

    auto work = edgedit::ggml_comm::all_gather_async(group, input, output);
    work->wait();

    std::vector<float> got = tensor_get_f32(output);

    std::vector<float> expected;
    for (int r = 0; r < world_size; ++r) {
        expected.push_back(static_cast<float>(r + 1));
        expected.push_back(static_cast<float>((r + 1) * 10));
    }

    check_close(got, expected, "ggml_comm async all_gather");

    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
}

void run_ggml_comm_tests(edgedit::parallel::ProcessGroup& group,
                         const Args& args) {
    std::cout
        << "[rank " << group.rank() << "] run ggml_comm tests"
        << " process_group_backend=" << args.backend
        << " local_rank=" << args.local_rank
        << "\n";

    ggml_backend_t backend = init_ggml_backend_for_test(args);

    std::cout
        << "[rank " << group.rank() << "] ggml backend = "
        << ggml_backend_name(backend)
        << "\n";

    run_all_reduce_test(group, backend);
    run_all_gather_test(group, backend);
    run_all_to_all_test(group, backend);
    run_broadcast_test(group, backend);
    run_async_all_gather_test(group, backend);

    ggml_backend_free(backend);
}

} // namespace

int main(int argc, char** argv) {
    try {
        Args args = parse_args(argc, argv);

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

        if (group->backend() == Backend::kCpu || group->backend() == Backend::kNccl) {
            run_ggml_comm_tests(*group, args);
        } else {
            throw std::runtime_error("unsupported process group backend");
        }

        group->barrier();

        if (group->rank() == 0) {
            std::cout << "ggml_comm_test passed\n";
        }

        return 0;
    } catch (const std::exception& e) {
        std::cerr << "ggml_comm_test failed: " << e.what() << "\n";
        return 1;
    }
}