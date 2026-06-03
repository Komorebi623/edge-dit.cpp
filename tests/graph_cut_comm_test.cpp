#include "backend/ggml/ggml_graph_cut.h"
#include "backend/ggml/ggml_extend.hpp"
#include "parallel/process_group.hpp"

#include "ggml.h"
#include "ggml-backend.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <initializer_list>
#include <iostream>
#include <memory>
#include <optional>
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

std::string env_str_any(std::initializer_list<const char*> names,
                        const std::string& fallback = "") {
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
                << "Usage: graph-cut-comm-test [options]\n"
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

        const std::string cuda_name = "CUDA" + std::to_string(args.local_rank);
        ggml_backend_t backend = ggml_backend_init_by_name(cuda_name.c_str(), nullptr);

        if (backend == nullptr) {
            std::ostringstream oss;
            oss << "failed to init ggml CUDA backend by name: " << cuda_name
                << ". Make sure build enables -DED_GGML_CUDA=ON.";
            throw std::runtime_error(oss.str());
        }

        return backend;
    }

    throw std::runtime_error("unsupported test backend: " + args.backend);
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

ggml_tensor* new_1d_f32(ggml_context* ctx,
                        int64_t n,
                        const char* name) {
    ggml_tensor* tensor = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n);
    ggml_set_name(tensor, name);
    return tensor;
}

void tensor_set_f32(ggml_tensor* tensor,
                    const std::vector<float>& values) {
    if (static_cast<size_t>(ggml_nelements(tensor)) != values.size()) {
        throw std::runtime_error("tensor_set_f32 size mismatch");
    }
    ggml_backend_tensor_set(tensor, values.data(), 0, values.size() * sizeof(float));
}

std::vector<float> tensor_get_f32(ggml_tensor* tensor) {
    std::vector<float> values(static_cast<size_t>(ggml_nelements(tensor)));
    ggml_backend_tensor_get(tensor, values.data(), 0, values.size() * sizeof(float));
    return values;
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

int find_node_index(ggml_cgraph* gf,
                    const ggml_tensor* tensor,
                    const char* debug_name) {
    const int n_nodes = ggml_graph_n_nodes(gf);
    for (int i = 0; i < n_nodes; ++i) {
        if (ggml_graph_node(gf, i) == tensor) {
            return i;
        }
    }

    std::ostringstream oss;
    oss << "failed to find graph node index for " << debug_name;
    throw std::runtime_error(oss.str());
}

void run_graph(ggml_backend_t backend,
               ggml_cgraph* gf) {
    ggml_status status = ggml_backend_graph_compute(backend, gf);
    if (status != GGML_STATUS_SUCCESS) {
        std::ostringstream oss;
        oss << "ggml_backend_graph_compute failed: " << ggml_status_to_string(status);
        throw std::runtime_error(oss.str());
    }
    ggml_backend_synchronize(backend);
}

void execute_segment_comm_or_throw(edgedit::parallel::ProcessGroup& group,
                                   ggml_cgraph* gf,
                                   const sd::ggml_graph_cut::Segment& segment,
                                   const char* test_name) {
    std::string error;
    if (!sd::ggml_graph_cut::execute_segment_comm_ops(group, gf, segment, &error)) {
        std::ostringstream oss;
        oss << test_name << " execute_segment_comm_ops failed: " << error;
        throw std::runtime_error(oss.str());
    }
}

void run_all_reduce_segment_comm_test(edgedit::parallel::ProcessGroup& group,
                                      ggml_backend_t backend) {
    const int rank = group.rank();
    const int world_size = group.size();

    ggml_context* ctx = new_test_context();

    ggml_tensor* input_leaf = new_1d_f32(ctx, 2, "all_reduce_input_leaf");
    ggml_tensor* output_leaf = new_1d_f32(ctx, 2, "all_reduce_output_leaf");

    ggml_tensor* input_node = ggml_dup(ctx, input_leaf);
    ggml_set_name(input_node, "all_reduce_input_node");

    ggml_tensor* output_node = ggml_dup(ctx, output_leaf);
    ggml_set_name(output_node, "all_reduce_output_node");

    ggml_cgraph* gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, input_node);
    ggml_build_forward_expand(gf, output_node);

    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (buffer == nullptr) {
        ggml_free(ctx);
        throw std::runtime_error("alloc ctx tensors failed in all_reduce test");
    }

    tensor_set_f32(input_leaf, {
        static_cast<float>(rank + 1),
        static_cast<float>((rank + 1) * 10),
    });
    tensor_set_f32(output_leaf, {0.0f, 0.0f});

    run_graph(backend, gf);

    const int input_idx = find_node_index(gf, input_node, "all_reduce_input_node");
    const int output_idx = find_node_index(gf, output_node, "all_reduce_output_node");

    sd::ggml_graph_cut::Segment segment;
    segment.group_name = "graph_cut_comm_all_reduce";

    sd::ggml_graph_cut::Segment::CommOp op;
    op.kind = sd::ggml_graph_cut::Segment::CommKind::ALL_REDUCE;
    op.name = "all_reduce_smoke";
    op.input_node_index = input_idx;
    op.output_node_index = output_idx;
    op.reduce_op = ReduceOp::kSum;
    segment.comm_ops.push_back(op);

    execute_segment_comm_or_throw(group, gf, segment, "all_reduce");

    const float sum_rank = static_cast<float>(world_size * (world_size + 1) / 2);
    check_close(tensor_get_f32(output_node),
                {sum_rank, sum_rank * 10.0f},
                "graph_cut all_reduce");

    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
}

void run_all_gather_segment_comm_test(edgedit::parallel::ProcessGroup& group,
                                      ggml_backend_t backend) {
    const int rank = group.rank();
    const int world_size = group.size();

    ggml_context* ctx = new_test_context();

    ggml_tensor* input_leaf = new_1d_f32(ctx, 2, "all_gather_input_leaf");
    ggml_tensor* output_leaf = new_1d_f32(ctx, 2 * world_size, "all_gather_output_leaf");

    ggml_tensor* input_node = ggml_dup(ctx, input_leaf);
    ggml_set_name(input_node, "all_gather_input_node");

    ggml_tensor* output_node = ggml_dup(ctx, output_leaf);
    ggml_set_name(output_node, "all_gather_output_node");

    ggml_cgraph* gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, input_node);
    ggml_build_forward_expand(gf, output_node);

    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (buffer == nullptr) {
        ggml_free(ctx);
        throw std::runtime_error("alloc ctx tensors failed in all_gather test");
    }

    tensor_set_f32(input_leaf, {
        static_cast<float>(rank + 1),
        static_cast<float>((rank + 1) * 10),
    });
    tensor_set_f32(output_leaf, std::vector<float>(static_cast<size_t>(2 * world_size), 0.0f));

    run_graph(backend, gf);

    const int input_idx = find_node_index(gf, input_node, "all_gather_input_node");
    const int output_idx = find_node_index(gf, output_node, "all_gather_output_node");

    sd::ggml_graph_cut::Segment segment;
    segment.group_name = "graph_cut_comm_all_gather";

    sd::ggml_graph_cut::Segment::CommOp op;
    op.kind = sd::ggml_graph_cut::Segment::CommKind::ALL_GATHER;
    op.name = "all_gather_smoke";
    op.input_node_index = input_idx;
    op.output_node_index = output_idx;
    segment.comm_ops.push_back(op);

    execute_segment_comm_or_throw(group, gf, segment, "all_gather");

    std::vector<float> expected;
    for (int r = 0; r < world_size; ++r) {
        expected.push_back(static_cast<float>(r + 1));
        expected.push_back(static_cast<float>((r + 1) * 10));
    }

    check_close(tensor_get_f32(output_node),
                expected,
                "graph_cut all_gather");

    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
}

void run_comm_annotation_to_plan_test(edgedit::parallel::ProcessGroup& group,
                                      ggml_backend_t backend) {
    const int rank = group.rank();
    const int world_size = group.size();

    ggml_context* ctx = new_test_context();

    ggml_tensor* input_leaf = new_1d_f32(ctx, 2, "annotation_input_leaf");
    ggml_tensor* output_leaf = new_1d_f32(ctx, 2 * world_size, "annotation_output_leaf");

    ggml_tensor* input_node = ggml_dup(ctx, input_leaf);
    ggml_set_name(input_node, "annotation_input_node");

    ggml_tensor* output_node = ggml_dup(ctx, output_leaf);
    ggml_set_name(output_node, "annotation_output_node");

    ggml_cgraph* gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, input_node);
    ggml_build_forward_expand(gf, output_node);

    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (buffer == nullptr) {
        ggml_free(ctx);
        throw std::runtime_error("alloc ctx tensors failed in comm annotation test");
    }

    tensor_set_f32(input_leaf, {
        static_cast<float>(rank + 1),
        static_cast<float>((rank + 1) * 10),
    });
    tensor_set_f32(output_leaf, std::vector<float>(static_cast<size_t>(2 * world_size), 0.0f));

    run_graph(backend, gf);

    const int input_idx = find_node_index(gf, input_node, "annotation_input_node");
    const int output_idx = find_node_index(gf, output_node, "annotation_output_node");

    sd::ggml_graph_cut::Plan base_plan;
    base_plan.available = true;
    base_plan.valid = true;
    base_plan.has_cuts = true;
    base_plan.n_nodes = ggml_graph_n_nodes(gf);
    base_plan.n_leafs = sd::ggml_graph_cut::leaf_count(gf);

    sd::ggml_graph_cut::Segment segment;
    segment.group_name = "annotation_segment";
    segment.internal_node_indices.push_back(input_idx);
    segment.internal_node_indices.push_back(output_idx);
    segment.output_node_indices.push_back(output_idx);
    base_plan.segments.push_back(segment);

    sd::ggml_graph_cut::clear_comm_marks();
    sd::ggml_graph_cut::mark_comm_op(input_node,
                                     output_node,
                                     sd::ggml_graph_cut::Segment::CommKind::ALL_GATHER,
                                     "annotation_all_gather");

    sd::ggml_graph_cut::Plan annotated_plan =
        sd::ggml_graph_cut::attach_comm_ops_to_plan(gf,
                                                    base_plan,
                                                    "graph_cut_comm_test");
    sd::ggml_graph_cut::clear_comm_marks();

    if (annotated_plan.segments.size() != 1) {
        throw std::runtime_error("comm annotation test expected one segment");
    }
    const auto& annotated_segment = annotated_plan.segments.front();
    if (annotated_segment.comm_ops.size() != 1) {
        throw std::runtime_error("comm annotation test expected one generated comm op");
    }
    const auto& op = annotated_segment.comm_ops.front();
    if (op.kind != sd::ggml_graph_cut::Segment::CommKind::ALL_GATHER ||
        op.input_node_index != input_idx ||
        op.output_node_index != output_idx ||
        op.name != "annotation_all_gather") {
        throw std::runtime_error("comm annotation test generated wrong comm op");
    }

    execute_segment_comm_or_throw(group, gf, annotated_segment, "comm_annotation_to_plan");

    std::vector<float> expected;
    for (int r = 0; r < world_size; ++r) {
        expected.push_back(static_cast<float>(r + 1));
        expected.push_back(static_cast<float>((r + 1) * 10));
    }

    check_close(tensor_get_f32(output_node),
                expected,
                "graph_cut comm annotation all_gather");

    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
}

void run_all_to_all_segment_comm_test(edgedit::parallel::ProcessGroup& group,
                                      ggml_backend_t backend) {
    const int rank = group.rank();
    const int world_size = group.size();
    const size_t count_per_peer = 1;

    ggml_context* ctx = new_test_context();

    ggml_tensor* input_leaf = new_1d_f32(ctx, world_size, "all_to_all_input_leaf");
    ggml_tensor* output_leaf = new_1d_f32(ctx, world_size, "all_to_all_output_leaf");

    ggml_tensor* input_node = ggml_dup(ctx, input_leaf);
    ggml_set_name(input_node, "all_to_all_input_node");

    ggml_tensor* output_node = ggml_dup(ctx, output_leaf);
    ggml_set_name(output_node, "all_to_all_output_node");

    ggml_cgraph* gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, input_node);
    ggml_build_forward_expand(gf, output_node);

    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (buffer == nullptr) {
        ggml_free(ctx);
        throw std::runtime_error("alloc ctx tensors failed in all_to_all test");
    }

    std::vector<float> input_values(static_cast<size_t>(world_size));
    for (int peer = 0; peer < world_size; ++peer) {
        input_values[static_cast<size_t>(peer)] =
            static_cast<float>(rank * 100 + peer);
    }

    tensor_set_f32(input_leaf, input_values);
    tensor_set_f32(output_leaf, std::vector<float>(static_cast<size_t>(world_size), 0.0f));

    run_graph(backend, gf);

    const int input_idx = find_node_index(gf, input_node, "all_to_all_input_node");
    const int output_idx = find_node_index(gf, output_node, "all_to_all_output_node");

    sd::ggml_graph_cut::Segment segment;
    segment.group_name = "graph_cut_comm_all_to_all";

    sd::ggml_graph_cut::Segment::CommOp op;
    op.kind = sd::ggml_graph_cut::Segment::CommKind::ALL_TO_ALL;
    op.name = "all_to_all_smoke";
    op.input_node_index = input_idx;
    op.output_node_index = output_idx;
    op.count_per_peer = count_per_peer;
    segment.comm_ops.push_back(op);

    execute_segment_comm_or_throw(group, gf, segment, "all_to_all");

    std::vector<float> expected;
    for (int src_rank = 0; src_rank < world_size; ++src_rank) {
        expected.push_back(static_cast<float>(src_rank * 100 + rank));
    }

    check_close(tensor_get_f32(output_node),
                expected,
                "graph_cut all_to_all");

    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
}

void run_broadcast_segment_comm_test(edgedit::parallel::ProcessGroup& group,
                                     ggml_backend_t backend) {
    const int rank = group.rank();

    ggml_context* ctx = new_test_context();

    ggml_tensor* input_leaf = new_1d_f32(ctx, 2, "broadcast_input_leaf");

    ggml_tensor* input_node = ggml_dup(ctx, input_leaf);
    ggml_set_name(input_node, "broadcast_input_node");

    ggml_cgraph* gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, input_node);

    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (buffer == nullptr) {
        ggml_free(ctx);
        throw std::runtime_error("alloc ctx tensors failed in broadcast test");
    }

    if (rank == 0) {
        tensor_set_f32(input_leaf, {123.0f, 456.0f});
    } else {
        tensor_set_f32(input_leaf, {0.0f, 0.0f});
    }

    run_graph(backend, gf);

    const int input_idx = find_node_index(gf, input_node, "broadcast_input_node");

    sd::ggml_graph_cut::Segment segment;
    segment.group_name = "graph_cut_comm_broadcast";

    sd::ggml_graph_cut::Segment::CommOp op;
    op.kind = sd::ggml_graph_cut::Segment::CommKind::BROADCAST;
    op.name = "broadcast_smoke";
    op.input_node_index = input_idx;
    op.output_node_index = -1;
    op.root = 0;
    segment.comm_ops.push_back(op);

    execute_segment_comm_or_throw(group, gf, segment, "broadcast");

    check_close(tensor_get_f32(input_node),
                {123.0f, 456.0f},
                "graph_cut broadcast");

    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
}

std::vector<float> sd_tensor_to_vector(const sd::Tensor<float>& tensor) {
    if (tensor.empty()) {
        throw std::runtime_error("sd_tensor_to_vector got empty tensor");
    }

    std::vector<float> values(static_cast<size_t>(tensor.numel()));
    std::copy(tensor.data(), tensor.data() + tensor.numel(), values.begin());
    return values;
}

std::shared_ptr<edgedit::parallel::ProcessGroup>
make_non_owning_process_group_ref(edgedit::parallel::ProcessGroup& group) {
    return std::shared_ptr<edgedit::parallel::ProcessGroup>(
        &group,
        [](edgedit::parallel::ProcessGroup*) {
            // non-owning reference, do not delete
        }
    );
}

class GraphCutHookTestRunner final : public GGMLRunner {
public:
    explicit GraphCutHookTestRunner(ggml_backend_t backend)
        : GGMLRunner(backend, false) {}

    std::string get_desc() override {
        return "GraphCutHookTestRunner";
    }

    std::optional<sd::Tensor<float>> run_all_gather_hook_test(
        edgedit::parallel::ProcessGroup& group
    ) {
        set_process_group(make_non_owning_process_group_ref(group));

        reset_compute_ctx();

        const int rank = group.rank();
        const int world_size = group.size();

        std::vector<float> local_values = {
            static_cast<float>(rank + 1),
            static_cast<float>((rank + 1) * 10),
        };

        std::vector<float> gather_init_values(
            static_cast<size_t>(2 * world_size),
            0.0f
        );

        // leaf: execute_graph() 会通过 set_backend_tensor_data() 把 host 数据拷入 backend tensor
        ggml_tensor* local_leaf = new_1d_f32(compute_ctx, 2, "runner_hook_local_leaf");
        set_backend_tensor_data(local_leaf, local_values.data());

        // node: graph compute 后得到本 rank 的 local tensor，作为 all_gather 输入
        ggml_tensor* local_node = ggml_dup(compute_ctx, local_leaf);
        ggml_set_name(local_node, "runner_hook_local_node");

        // leaf: 初始化 all_gather 输出 buffer 为 0
        ggml_tensor* gather_leaf = new_1d_f32(
            compute_ctx,
            2 * world_size,
            "runner_hook_gather_leaf"
        );
        set_backend_tensor_data(gather_leaf, gather_init_values.data());

        // node: graph compute 先写入 0，post_compute_cb 再用 all_gather 覆盖
        ggml_tensor* gather_node = ggml_dup(compute_ctx, gather_leaf);

        // execute_graph() 最后会按 final_result_name 读取返回值
        ggml_set_name(gather_node, final_result_name.c_str());

        ggml_cgraph* gf = new_graph_custom(16);
        ggml_build_forward_expand(gf, local_node);
        ggml_build_forward_expand(gf, gather_node);

        const int local_idx = find_node_index(gf, local_node, "runner_hook_local_node");
        const int gather_idx = find_node_index(gf, gather_node, "runner_hook_gather_node");

        sd::ggml_graph_cut::Segment segment;
        segment.group_name = "runner_hook_segment";

        sd::ggml_graph_cut::Segment::CommOp op;
        op.kind = sd::ggml_graph_cut::Segment::CommKind::ALL_GATHER;
        op.name = "runner_hook_all_gather";
        op.input_node_index = local_idx;
        op.output_node_index = gather_idx;
        segment.comm_ops.push_back(op);

        post_compute_cb_t post_compute_cb = [this, &group, gf, segment](ggml_cgraph*) -> bool {
            std::string comm_error;
            if (!sd::ggml_graph_cut::execute_segment_comm_ops(group, gf, segment, &comm_error)) {
                std::cerr
                    << "runner hook execute_segment_comm_ops failed: "
                    << comm_error
                    << std::endl;
                return false;
            }
            return true;
        };

        // 关键：这里直接测试 GGMLRunner::execute_graph() 的 post_compute_cb。
        //
        // 执行顺序是：
        //   ggml_backend_graph_compute()
        //   post_compute_cb -> execute_segment_comm_ops()
        //   copy_cache_tensors_to_cache_buffer()
        //   read final_result_name
        //
        // 不走 compute_with_graph_cuts()，避免手工 Plan 不满足 build_segment_graph() 的完整约束。
        return execute_graph<float>(
            gf,
            1,
            true,
            {},
            false,
            false,
            nullptr,
            post_compute_cb
        );
    }
};
void run_runner_hook_smoke_test(edgedit::parallel::ProcessGroup& group,
                                ggml_backend_t backend) {
    if (!group.enabled()) {
        return;
    }

    std::cout
        << "[rank " << group.rank()
        << "] run GGMLRunner graph-cut hook smoke test\n";

    GraphCutHookTestRunner runner(backend);
    std::optional<sd::Tensor<float>> output =
        runner.run_all_gather_hook_test(group);

    if (!output.has_value()) {
        throw std::runtime_error("runner hook smoke test returned nullopt");
    }

    std::vector<float> got = sd_tensor_to_vector(*output);

    std::vector<float> expected;
    for (int r = 0; r < group.size(); ++r) {
        expected.push_back(static_cast<float>(r + 1));
        expected.push_back(static_cast<float>((r + 1) * 10));
    }

    check_close(got, expected, "GGMLRunner graph-cut hook all_gather");
}

void run_graph_cut_comm_tests(edgedit::parallel::ProcessGroup& group,
                              const Args& args) {
    std::cout
        << "[rank " << group.rank() << "] run graph_cut comm tests"
        << " backend=" << args.backend
        << " local_rank=" << args.local_rank
        << "\n";

    ggml_backend_t backend = init_ggml_backend_for_test(args);

    std::cout
        << "[rank " << group.rank() << "] ggml backend = "
        << ggml_backend_name(backend)
        << "\n";

    run_all_reduce_segment_comm_test(group, backend);
    run_all_gather_segment_comm_test(group, backend);
    run_all_to_all_segment_comm_test(group, backend);
    run_broadcast_segment_comm_test(group, backend);
    run_comm_annotation_to_plan_test(group, backend);

    // 新增：验证 GGMLRunner::compute_with_graph_cuts()
    // 是否真的会通过 post_compute_cb 触发 segment.comm_ops。
    run_runner_hook_smoke_test(group, backend);

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

        if (group->backend() != Backend::kCpu &&
            group->backend() != Backend::kNccl) {
            throw std::runtime_error("unsupported process group backend");
        }

        run_graph_cut_comm_tests(*group, args);

        group->barrier();

        if (group->rank() == 0) {
            std::cout << "graph_cut_comm_test passed\n";
        }

        return 0;
    } catch (const std::exception& e) {
        std::cerr << "graph_cut_comm_test failed: " << e.what() << "\n";
        return 1;
    }
}
