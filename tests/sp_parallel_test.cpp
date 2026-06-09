#include "backend/ggml/ggml_graph_cut.h"
#include "backend/ggml/ggml_extend.hpp"
#include "parallel/process_group.hpp"
#include "parallel/sp_parallel.hpp"

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
#include <unordered_set>
#include <vector>

using edgedit::parallel::Backend;
using edgedit::parallel::ParallelConfig;

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
                << "Usage: sp-parallel-test [options]\n"
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

ggml_backend_t init_backend(const Args& args) {
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
    params.mem_size = 32 * 1024 * 1024;
    params.mem_buffer = nullptr;
    params.no_alloc = true;

    ggml_context* ctx = ggml_init(params);
    if (ctx == nullptr) {
        throw std::runtime_error("ggml_init failed");
    }
    return ctx;
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
        }
    );
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

template <typename Fn>
void expect_throws(Fn&& fn,
                   const char* name) {
    bool threw = false;
    try {
        fn();
    } catch (const std::exception&) {
        threw = true;
    }

    if (!threw) {
        throw std::runtime_error(std::string(name) + " expected an exception");
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

sd::ggml_graph_cut::Plan make_single_segment_plan(ggml_cgraph* gf,
                                                  std::initializer_list<int> internal_nodes,
                                                  std::initializer_list<int> output_nodes) {
    sd::ggml_graph_cut::Plan plan;
    plan.available = true;
    plan.valid = true;
    plan.has_cuts = true;
    plan.n_nodes = ggml_graph_n_nodes(gf);
    plan.n_leafs = sd::ggml_graph_cut::leaf_count(gf);

    sd::ggml_graph_cut::Segment segment;
    segment.group_name = "sp_parallel_test_segment";
    for (int node : internal_nodes) {
        segment.internal_node_indices.push_back(node);
    }
    for (int node : output_nodes) {
        segment.output_node_indices.push_back(node);
    }
    plan.segments.push_back(std::move(segment));
    return plan;
}

sd::ggml_graph_cut::Segment attach_one_comm_or_throw(ggml_cgraph* gf,
                                                    const sd::ggml_graph_cut::Plan& base_plan,
                                                    const char* test_name) {
    sd::ggml_graph_cut::Plan annotated =
        sd::ggml_graph_cut::attach_comm_ops_to_plan(gf, base_plan, test_name);
    sd::ggml_graph_cut::clear_comm_marks();

    if (annotated.segments.size() != 1) {
        throw std::runtime_error(std::string(test_name) + " expected one segment");
    }
    const auto& segment = annotated.segments.front();
    if (segment.comm_ops.size() != 1) {
        throw std::runtime_error(std::string(test_name) + " expected one comm op");
    }
    return segment;
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

float sequence_value(int token,
                     int hidden) {
    return static_cast<float>(token * 10 + hidden + 1);
}

std::vector<float> make_sequence_full_values(int hidden,
                                             int seq) {
    std::vector<float> values;
    values.reserve(static_cast<size_t>(hidden * seq));
    for (int token = 0; token < seq; ++token) {
        for (int h = 0; h < hidden; ++h) {
            values.push_back(sequence_value(token, h));
        }
    }
    return values;
}

std::vector<float> make_sequence_padded_expected(int hidden,
                                                 int original_seq,
                                                 int local_seq,
                                                 int world_size) {
    std::vector<float> expected;
    expected.reserve(static_cast<size_t>(hidden * local_seq * world_size));
    for (int rank = 0; rank < world_size; ++rank) {
        for (int local = 0; local < local_seq; ++local) {
            const int token = rank * local_seq + local;
            for (int h = 0; h < hidden; ++h) {
                expected.push_back(token < original_seq ? sequence_value(token, h) : 0.0f);
            }
        }
    }
    return expected;
}

std::vector<float> make_sequence_local_expected(int rank,
                                                int hidden,
                                                int original_seq,
                                                int local_seq) {
    std::vector<float> expected;
    expected.reserve(static_cast<size_t>(hidden * local_seq));
    for (int local = 0; local < local_seq; ++local) {
        const int token = rank * local_seq + local;
        for (int h = 0; h < hidden; ++h) {
            expected.push_back(token < original_seq ? sequence_value(token, h) : 0.0f);
        }
    }
    return expected;
}

float a2a_seq_to_head_value(int src_rank,
                            int local_seq,
                            int full_head,
                            int hidden) {
    return static_cast<float>(src_rank * 1000 + local_seq * 100 + full_head * 10 + hidden + 1);
}

float a2a_head_to_seq_value(int src_rank,
                            int global_seq,
                            int local_head,
                            int hidden) {
    return static_cast<float>(src_rank * 1000 + global_seq * 100 + local_head * 10 + hidden + 1);
}

void run_validation_tests(edgedit::parallel::ProcessGroup& group) {
    const int rank = group.rank();
    const int world_size = group.size();
    if (world_size != 2) {
        throw std::runtime_error("run_validation_tests expects world_size == 2");
    }

    if (edgedit::parallel::sp_sequence_padding(7, world_size) != 1 ||
        edgedit::parallel::sp_sequence_padding(8, world_size) != 0 ||
        edgedit::parallel::sp_sequence_padding(0, world_size) != 0) {
        throw std::runtime_error("sp_sequence_padding returned wrong values");
    }
    expect_throws([]() {
        (void)edgedit::parallel::sp_sequence_padding(8, 0);
    }, "sp_sequence_padding invalid world_size");

    ggml_context* ctx = new_test_context();
    ggml_tensor* sequence = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 2, 8, 1, 1);
    ggml_tensor* local = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 2, 4, 1, 1);
    ggml_tensor* local_batch_gt_one = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 2, 4, 2, 1);
    ggml_tensor* local_ne3_gt_one = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 2, 4, 1, 2);
    ggml_tensor* seq_to_head_bad_heads = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 2, 3, 4, 1);
    ggml_tensor* seq_to_head_bad_batch = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 2, 4, 4, 2);
    ggml_tensor* head_to_seq_bad_sequence = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 2, 2, 5, 1);
    ggml_tensor* head_to_seq_bad_batch = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 2, 2, 6, 2);

    expect_throws([&]() {
        (void)edgedit::parallel::sp_split_sequence(ctx, sequence, world_size, world_size, 1);
    }, "sp_split_sequence invalid rank");
    expect_throws([&]() {
        (void)edgedit::parallel::sp_split_sequence_view(ctx, sequence, rank, world_size, 0);
    }, "sp_split_sequence_view invalid seq_dim");
    expect_throws([&]() {
        (void)edgedit::parallel::sp_mark_gather_sequence(ctx, local, world_size, 1, 8);
    }, "sp_mark_gather_sequence invalid pad");
    expect_throws([&]() {
        (void)edgedit::parallel::sp_mark_gather_sequence(ctx, local_batch_gt_one, world_size, 1, 0);
    }, "sp_mark_gather_sequence batch greater than one");
    expect_throws([&]() {
        (void)edgedit::parallel::sp_mark_gather_sequence(ctx, local_ne3_gt_one, world_size, 1, 0);
    }, "sp_mark_gather_sequence ne3 greater than one");
    expect_throws([&]() {
        (void)edgedit::parallel::sp_all_to_all_4d_seq_to_head(ctx, seq_to_head_bad_heads, world_size);
    }, "sp_all_to_all_4d_seq_to_head non-divisible heads");
    expect_throws([&]() {
        (void)edgedit::parallel::sp_all_to_all_4d_seq_to_head(ctx, seq_to_head_bad_batch, world_size);
    }, "sp_all_to_all_4d_seq_to_head batch greater than one");
    expect_throws([&]() {
        (void)edgedit::parallel::sp_all_to_all_4d_head_to_seq(ctx, head_to_seq_bad_sequence, world_size);
    }, "sp_all_to_all_4d_head_to_seq non-divisible sequence");
    expect_throws([&]() {
        (void)edgedit::parallel::sp_all_to_all_4d_head_to_seq(ctx, head_to_seq_bad_batch, world_size);
    }, "sp_all_to_all_4d_head_to_seq batch greater than one");

    ggml_free(ctx);
}

void run_split_gather_test(edgedit::parallel::ProcessGroup& group,
                           ggml_backend_t backend) {
    const int rank = group.rank();
    const int world_size = group.size();
    if (world_size != 2) {
        throw std::runtime_error("run_split_gather_test expects world_size == 2");
    }

    ggml_context* ctx = new_test_context();

    const int hidden = 2;
    const int seq = 7;
    ggml_tensor* full = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, hidden, seq, 1, 1);
    ggml_set_name(full, "sp_sequence_full");

    sd::ggml_graph_cut::clear_comm_marks();
    edgedit::parallel::SPSequenceSplit split =
        edgedit::parallel::sp_split_sequence(ctx,
                                             full,
                                             rank,
                                             world_size,
                                             1,
                                             "sp_test_sequence_split");

    if (split.pad != 1 ||
        split.original_seq_len != 7 ||
        split.padded_seq_len != 8 ||
        split.local_seq_len != 4 ||
        split.local->ne[0] != hidden ||
        split.local->ne[1] != 4) {
        throw std::runtime_error("sp_split_sequence returned wrong metadata or shape");
    }

    edgedit::parallel::SPSequenceGather gather =
        edgedit::parallel::sp_mark_gather_sequence(ctx,
                                                   split.local,
                                                   world_size,
                                                   1,
                                                   split.pad,
                                                   "sp_test_gather_sequence");

    ggml_cgraph* pre_graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(pre_graph, split.local);
    ggml_build_forward_expand(pre_graph, gather.recv);

    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (buffer == nullptr) {
        ggml_free(ctx);
        throw std::runtime_error("alloc ctx tensors failed in split/gather test");
    }

    tensor_set_f32(full, make_sequence_full_values(hidden, seq));
    tensor_set_f32(gather.recv->src[0],
                   std::vector<float>(static_cast<size_t>(ggml_nelements(gather.recv)), 0.0f));

    run_graph(backend, pre_graph);

    check_close(tensor_get_f32(split.local),
                make_sequence_local_expected(rank, hidden, seq, 4),
                "sp split padded local shard");

    const int local_idx = find_node_index(pre_graph, split.local, "split.local");
    const int recv_idx = find_node_index(pre_graph, gather.recv, "gather.recv");

    sd::ggml_graph_cut::Plan base_plan =
        make_single_segment_plan(pre_graph, {local_idx, recv_idx}, {recv_idx});
    sd::ggml_graph_cut::Segment segment =
        attach_one_comm_or_throw(pre_graph, base_plan, "sp_split_gather");

    if (segment.comm_ops.front().kind != sd::ggml_graph_cut::Segment::CommKind::ALL_GATHER) {
        throw std::runtime_error("sp_split_gather expected ALL_GATHER comm op");
    }

    execute_segment_comm_or_throw(group, pre_graph, segment, "sp_split_gather");

    std::vector<float> expected_padded =
        make_sequence_padded_expected(hidden, seq, 4, world_size);
    check_close(tensor_get_f32(gather.recv),
                expected_padded,
                "sp split local all_gather padded recv");

    // Validate the next-segment post layout. Because recv is a dup
    // node for graph membership, seed its leaf with the communication result
    // before computing the post-layout graph.
    tensor_set_f32(gather.recv->src[0], expected_padded);

    ggml_cgraph* post_graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(post_graph, gather.gathered);
    run_graph(backend, post_graph);

    check_close(tensor_get_f32(gather.gathered),
                make_sequence_full_values(hidden, seq),
                "sp gather post layout");

    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
}

void run_split_gather_no_pad_test(edgedit::parallel::ProcessGroup& group,
                                  ggml_backend_t backend) {
    const int rank = group.rank();
    const int world_size = group.size();
    if (world_size != 2) {
        throw std::runtime_error("run_split_gather_no_pad_test expects world_size == 2");
    }

    ggml_context* ctx = new_test_context();

    const int hidden = 3;
    const int seq = 8;
    ggml_tensor* full = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, hidden, seq, 1, 1);
    ggml_set_name(full, "sp_sequence_no_pad_full");

    sd::ggml_graph_cut::clear_comm_marks();
    edgedit::parallel::SPSequenceSplit split =
        edgedit::parallel::sp_split_sequence(ctx,
                                             full,
                                             rank,
                                             world_size,
                                             1,
                                             "sp_test_sequence_no_pad_split");

    if (split.pad != 0 ||
        split.original_seq_len != seq ||
        split.padded_seq_len != seq ||
        split.local_seq_len != seq / world_size ||
        split.local->ne[0] != hidden ||
        split.local->ne[1] != seq / world_size) {
        throw std::runtime_error("sp_split_sequence no-pad returned wrong metadata or shape");
    }

    edgedit::parallel::SPSequenceGather gather =
        edgedit::parallel::sp_mark_gather_sequence(ctx,
                                                   split.local,
                                                   world_size,
                                                   1,
                                                   split.pad,
                                                   "sp_test_gather_sequence_no_pad");

    ggml_cgraph* pre_graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(pre_graph, split.local);
    ggml_build_forward_expand(pre_graph, gather.recv);

    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (buffer == nullptr) {
        ggml_free(ctx);
        throw std::runtime_error("alloc ctx tensors failed in split/gather no-pad test");
    }

    tensor_set_f32(full, make_sequence_full_values(hidden, seq));
    tensor_set_f32(gather.recv->src[0],
                   std::vector<float>(static_cast<size_t>(ggml_nelements(gather.recv)), 0.0f));

    run_graph(backend, pre_graph);

    check_close(tensor_get_f32(split.local),
                make_sequence_local_expected(rank, hidden, seq, seq / world_size),
                "sp split no-pad local shard");

    const int local_idx = find_node_index(pre_graph, split.local, "split.no_pad.local");
    const int recv_idx = find_node_index(pre_graph, gather.recv, "gather.no_pad.recv");
    sd::ggml_graph_cut::Plan base_plan =
        make_single_segment_plan(pre_graph, {local_idx, recv_idx}, {recv_idx});
    sd::ggml_graph_cut::Segment segment =
        attach_one_comm_or_throw(pre_graph, base_plan, "sp_split_gather_no_pad");

    if (segment.comm_ops.front().kind != sd::ggml_graph_cut::Segment::CommKind::ALL_GATHER) {
        throw std::runtime_error("sp_split_gather_no_pad expected ALL_GATHER comm op");
    }

    execute_segment_comm_or_throw(group, pre_graph, segment, "sp_split_gather_no_pad");

    std::vector<float> expected = make_sequence_full_values(hidden, seq);
    check_close(tensor_get_f32(gather.recv),
                expected,
                "sp split no-pad all_gather recv");

    tensor_set_f32(gather.recv->src[0], expected);
    ggml_cgraph* post_graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(post_graph, gather.gathered);
    run_graph(backend, post_graph);

    check_close(tensor_get_f32(gather.gathered),
                expected,
                "sp gather no-pad post layout");

    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
}

void run_split_gather_batched_no_pad_test(edgedit::parallel::ProcessGroup& group,
                                          ggml_backend_t backend) {
    const int rank = group.rank();
    const int world_size = group.size();
    if (world_size != 2) {
        throw std::runtime_error("run_split_gather_batched_no_pad_test expects world_size == 2");
    }

    ggml_context* ctx = new_test_context();

    const int hidden = 3;
    const int txt_seq = 8;
    const int img_seq = 6;
    ggml_tensor* txt_full = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, hidden, txt_seq, 1, 1);
    ggml_tensor* img_full = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, hidden, img_seq, 1, 1);
    ggml_set_name(txt_full, "sp_sequence_batched_txt_full");
    ggml_set_name(img_full, "sp_sequence_batched_img_full");

    sd::ggml_graph_cut::clear_comm_marks();
    edgedit::parallel::SPSequenceSplit txt_split =
        edgedit::parallel::sp_split_sequence(ctx,
                                             txt_full,
                                             rank,
                                             world_size,
                                             1,
                                             "sp_test_sequence_batched_txt_split");
    edgedit::parallel::SPSequenceSplit img_split =
        edgedit::parallel::sp_split_sequence(ctx,
                                             img_full,
                                             rank,
                                             world_size,
                                             1,
                                             "sp_test_sequence_batched_img_split");

    edgedit::parallel::SPSequenceGatherBatch gather =
        edgedit::parallel::sp_mark_gather_sequence_batched(ctx,
                                                           {txt_split.local, img_split.local},
                                                           world_size,
                                                           1,
                                                           {txt_split.pad, img_split.pad},
                                                           "sp_test_gather_sequence_batched");

    if (gather.gathered.size() != 2 ||
        gather.gathered_padded.size() != 2 ||
        gather.count_per_rank != static_cast<size_t>(hidden * (txt_seq + img_seq) / world_size) ||
        gather.gathered[0]->ne[0] != hidden ||
        gather.gathered[0]->ne[1] != txt_seq ||
        gather.gathered[1]->ne[0] != hidden ||
        gather.gathered[1]->ne[1] != img_seq) {
        throw std::runtime_error("sp_mark_gather_sequence_batched returned wrong metadata or shape");
    }

    ggml_cgraph* pre_graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(pre_graph, txt_split.local);
    ggml_build_forward_expand(pre_graph, img_split.local);
    ggml_build_forward_expand(pre_graph, gather.recv_flat);

    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (buffer == nullptr) {
        ggml_free(ctx);
        throw std::runtime_error("alloc ctx tensors failed in batched split/gather test");
    }

    std::vector<float> txt_values = make_sequence_full_values(hidden, txt_seq);
    std::vector<float> img_values = make_sequence_full_values(hidden, img_seq);
    for (float& value : img_values) {
        value += 10000.0f;
    }
    tensor_set_f32(txt_full, txt_values);
    tensor_set_f32(img_full, img_values);
    tensor_set_f32(gather.recv_flat->src[0],
                   std::vector<float>(static_cast<size_t>(ggml_nelements(gather.recv_flat)), 0.0f));

    run_graph(backend, pre_graph);

    check_close(tensor_get_f32(txt_split.local),
                make_sequence_local_expected(rank, hidden, txt_seq, txt_seq / world_size),
                "sp split batched txt local shard");
    std::vector<float> img_local_expected =
        make_sequence_local_expected(rank, hidden, img_seq, img_seq / world_size);
    for (float& value : img_local_expected) {
        value += 10000.0f;
    }
    check_close(tensor_get_f32(img_split.local),
                img_local_expected,
                "sp split batched img local shard");

    const int send_idx = find_node_index(pre_graph, gather.send_flat, "gather.batched.send_flat");
    const int recv_idx = find_node_index(pre_graph, gather.recv_flat, "gather.batched.recv_flat");
    sd::ggml_graph_cut::Plan base_plan =
        make_single_segment_plan(pre_graph, {send_idx, recv_idx}, {recv_idx});
    sd::ggml_graph_cut::Segment segment =
        attach_one_comm_or_throw(pre_graph, base_plan, "sp_split_gather_batched_no_pad");

    if (segment.comm_ops.front().kind != sd::ggml_graph_cut::Segment::CommKind::ALL_GATHER) {
        throw std::runtime_error("sp_split_gather_batched_no_pad expected one ALL_GATHER comm op");
    }

    execute_segment_comm_or_throw(group, pre_graph, segment, "sp_split_gather_batched_no_pad");

    std::vector<float> expected_flat;
    expected_flat.reserve(static_cast<size_t>(hidden * (txt_seq + img_seq)));
    for (int src = 0; src < world_size; ++src) {
        std::vector<float> txt_local =
            make_sequence_local_expected(src, hidden, txt_seq, txt_seq / world_size);
        expected_flat.insert(expected_flat.end(), txt_local.begin(), txt_local.end());

        std::vector<float> img_local =
            make_sequence_local_expected(src, hidden, img_seq, img_seq / world_size);
        for (float& value : img_local) {
            value += 10000.0f;
        }
        expected_flat.insert(expected_flat.end(), img_local.begin(), img_local.end());
    }
    check_close(tensor_get_f32(gather.recv_flat),
                expected_flat,
                "sp split batched all_gather flat recv");

    tensor_set_f32(gather.recv_flat->src[0], expected_flat);
    ggml_cgraph* post_graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(post_graph, gather.gathered[0]);
    ggml_build_forward_expand(post_graph, gather.gathered[1]);
    run_graph(backend, post_graph);

    check_close(tensor_get_f32(gather.gathered[0]),
                txt_values,
                "sp gather batched txt post layout");
    check_close(tensor_get_f32(gather.gathered[1]),
                img_values,
                "sp gather batched img post layout");

    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
}

std::vector<float> make_seq_to_head_input(int rank,
                                          int hidden,
                                          int heads,
                                          int shard_seq) {
    std::vector<float> values;
    values.reserve(static_cast<size_t>(hidden * heads * shard_seq));
    for (int seq = 0; seq < shard_seq; ++seq) {
        for (int head = 0; head < heads; ++head) {
            for (int h = 0; h < hidden; ++h) {
                values.push_back(a2a_seq_to_head_value(rank, seq, head, h));
            }
        }
    }
    return values;
}

std::vector<float> make_seq_to_head_expected(int rank,
                                             int world_size,
                                             int hidden,
                                             int shard_heads,
                                             int shard_seq) {
    std::vector<float> expected;
    expected.reserve(static_cast<size_t>(hidden * shard_heads * shard_seq * world_size));
    for (int src = 0; src < world_size; ++src) {
        for (int seq = 0; seq < shard_seq; ++seq) {
            for (int local_head = 0; local_head < shard_heads; ++local_head) {
                const int full_head = rank * shard_heads + local_head;
                for (int h = 0; h < hidden; ++h) {
                    expected.push_back(a2a_seq_to_head_value(src, seq, full_head, h));
                }
            }
        }
    }
    return expected;
}

std::vector<float> make_seq_to_head_seq_major_expected(int rank,
                                                       int world_size,
                                                       int hidden,
                                                       int shard_heads,
                                                       int shard_seq) {
    std::vector<float> expected;
    expected.reserve(static_cast<size_t>(hidden * shard_heads * shard_seq * world_size));
    for (int local_head = 0; local_head < shard_heads; ++local_head) {
        const int full_head = rank * shard_heads + local_head;
        for (int src = 0; src < world_size; ++src) {
            for (int seq = 0; seq < shard_seq; ++seq) {
                for (int h = 0; h < hidden; ++h) {
                    expected.push_back(a2a_seq_to_head_value(src, seq, full_head, h));
                }
            }
        }
    }
    return expected;
}

void run_seq_to_head_test(edgedit::parallel::ProcessGroup& group,
                          ggml_backend_t backend) {
    const int rank = group.rank();
    const int world_size = group.size();
    if (world_size != 2) {
        throw std::runtime_error("run_seq_to_head_test expects world_size == 2");
    }

    ggml_context* ctx = new_test_context();

    const int hidden = 2;
    const int heads = 4;
    const int shard_heads = heads / world_size;
    const int shard_seq = 3;
    ggml_tensor* input = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, hidden, heads, shard_seq, 1);
    ggml_set_name(input, "sp_seq_to_head_input");

    sd::ggml_graph_cut::clear_comm_marks();
    edgedit::parallel::SPAllToAll4DLayout layout =
        edgedit::parallel::sp_all_to_all_4d_seq_to_head(ctx,
                                                        input,
                                                        world_size,
                                                        "sp_test_seq_to_head");

    if (layout.output->ne[0] != hidden ||
        layout.output->ne[1] != shard_heads ||
        layout.output->ne[2] != shard_seq * world_size ||
        layout.count_per_peer != static_cast<size_t>(hidden * shard_heads * shard_seq)) {
        throw std::runtime_error("sp_all_to_all_4d_seq_to_head returned wrong metadata or shape");
    }

    ggml_cgraph* pre_graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(pre_graph, layout.send_flat);
    ggml_build_forward_expand(pre_graph, layout.recv_flat);

    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (buffer == nullptr) {
        ggml_free(ctx);
        throw std::runtime_error("alloc ctx tensors failed in seq_to_head test");
    }

    tensor_set_f32(input, make_seq_to_head_input(rank, hidden, heads, shard_seq));
    tensor_set_f32(layout.recv_flat->src[0],
                   std::vector<float>(static_cast<size_t>(ggml_nelements(layout.recv_flat)), 0.0f));

    run_graph(backend, pre_graph);

    const int send_idx = find_node_index(pre_graph, layout.send_flat, "seq_to_head.send_flat");
    const int recv_idx = find_node_index(pre_graph, layout.recv_flat, "seq_to_head.recv_flat");
    sd::ggml_graph_cut::Plan base_plan =
        make_single_segment_plan(pre_graph, {send_idx, recv_idx}, {recv_idx});
    sd::ggml_graph_cut::Segment segment =
        attach_one_comm_or_throw(pre_graph, base_plan, "sp_seq_to_head");

    if (segment.comm_ops.front().kind != sd::ggml_graph_cut::Segment::CommKind::ALL_TO_ALL ||
        segment.comm_ops.front().count_per_peer != layout.count_per_peer) {
        throw std::runtime_error("sp_seq_to_head expected ALL_TO_ALL comm op with layout count_per_peer");
    }

    execute_segment_comm_or_throw(group, pre_graph, segment, "sp_seq_to_head");

    std::vector<float> expected =
        make_seq_to_head_expected(rank, world_size, hidden, shard_heads, shard_seq);
    check_close(tensor_get_f32(layout.recv_flat),
                expected,
                "sp seq_to_head flat recv");

    tensor_set_f32(layout.recv_flat->src[0], expected);
    ggml_cgraph* post_graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(post_graph, layout.output);
    run_graph(backend, post_graph);
    check_close(tensor_get_f32(layout.output),
                expected,
                "sp seq_to_head post layout");

    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
}

void run_seq_to_head_batched_test(edgedit::parallel::ProcessGroup& group,
                                  ggml_backend_t backend) {
    const int rank = group.rank();
    const int world_size = group.size();
    if (world_size != 2) {
        throw std::runtime_error("run_seq_to_head_batched_test expects world_size == 2");
    }

    ggml_context* ctx = new_test_context();

    const int hidden = 2;
    const int heads = 4;
    const int shard_heads = heads / world_size;
    const int shard_seq = 3;
    ggml_tensor* q = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, hidden, heads, shard_seq, 1);
    ggml_tensor* k = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, hidden, heads, shard_seq, 1);
    ggml_tensor* v = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, hidden, heads, shard_seq, 1);
    ggml_set_name(q, "sp_seq_to_head_batched_q");
    ggml_set_name(k, "sp_seq_to_head_batched_k");
    ggml_set_name(v, "sp_seq_to_head_batched_v");

    sd::ggml_graph_cut::clear_comm_marks();
    edgedit::parallel::SPAllToAll4DBatchLayout layout =
        edgedit::parallel::sp_all_to_all_4d_seq_to_head_batched(ctx,
                                                                {q, k, v},
                                                                world_size,
                                                                "sp_test_seq_to_head_batched");

    if (layout.outputs.size() != 3 ||
        layout.outputs[0]->ne[0] != hidden ||
        layout.outputs[0]->ne[1] != shard_heads ||
        layout.outputs[0]->ne[2] != shard_seq * world_size ||
        layout.count_per_peer != static_cast<size_t>(hidden * 3 * shard_heads * shard_seq)) {
        throw std::runtime_error("sp_all_to_all_4d_seq_to_head_batched returned wrong metadata or shape");
    }

    ggml_cgraph* pre_graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(pre_graph, layout.send_flat);
    ggml_build_forward_expand(pre_graph, layout.recv_flat);

    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (buffer == nullptr) {
        ggml_free(ctx);
        throw std::runtime_error("alloc ctx tensors failed in seq_to_head batched test");
    }

    std::vector<float> q_values = make_seq_to_head_input(rank, hidden, heads, shard_seq);
    std::vector<float> k_values = q_values;
    std::vector<float> v_values = q_values;
    for (float& value : k_values) {
        value += 10000.0f;
    }
    for (float& value : v_values) {
        value += 20000.0f;
    }
    tensor_set_f32(q, q_values);
    tensor_set_f32(k, k_values);
    tensor_set_f32(v, v_values);
    tensor_set_f32(layout.recv_flat->src[0],
                   std::vector<float>(static_cast<size_t>(ggml_nelements(layout.recv_flat)), 0.0f));

    run_graph(backend, pre_graph);

    const int send_idx = find_node_index(pre_graph, layout.send_flat, "seq_to_head_batched.send_flat");
    const int recv_idx = find_node_index(pre_graph, layout.recv_flat, "seq_to_head_batched.recv_flat");
    sd::ggml_graph_cut::Plan base_plan =
        make_single_segment_plan(pre_graph, {send_idx, recv_idx}, {recv_idx});
    sd::ggml_graph_cut::Segment segment =
        attach_one_comm_or_throw(pre_graph, base_plan, "sp_seq_to_head_batched");

    if (segment.comm_ops.front().kind != sd::ggml_graph_cut::Segment::CommKind::ALL_TO_ALL ||
        segment.comm_ops.front().count_per_peer != layout.count_per_peer) {
        throw std::runtime_error("sp_seq_to_head_batched expected one ALL_TO_ALL comm op with layout count_per_peer");
    }

    execute_segment_comm_or_throw(group, pre_graph, segment, "sp_seq_to_head_batched");

    std::vector<float> q_expected =
        make_seq_to_head_expected(rank, world_size, hidden, shard_heads, shard_seq);
    std::vector<float> k_expected = q_expected;
    std::vector<float> v_expected = q_expected;
    for (float& value : k_expected) {
        value += 10000.0f;
    }
    for (float& value : v_expected) {
        value += 20000.0f;
    }

    std::vector<float> expected_flat;
    expected_flat.reserve(q_expected.size() + k_expected.size() + v_expected.size());
    for (int src = 0; src < world_size; ++src) {
        for (int seq = 0; seq < shard_seq; ++seq) {
            for (int local_head = 0; local_head < shard_heads; ++local_head) {
                const int full_head = rank * shard_heads + local_head;
                for (int h = 0; h < hidden; ++h) {
                    expected_flat.push_back(a2a_seq_to_head_value(src, seq, full_head, h));
                }
                for (int h = 0; h < hidden; ++h) {
                    expected_flat.push_back(a2a_seq_to_head_value(src, seq, full_head, h) + 10000.0f);
                }
                for (int h = 0; h < hidden; ++h) {
                    expected_flat.push_back(a2a_seq_to_head_value(src, seq, full_head, h) + 20000.0f);
                }
            }
        }
    }
    check_close(tensor_get_f32(layout.recv_flat),
                expected_flat,
                "sp seq_to_head batched flat recv");

    tensor_set_f32(layout.recv_flat->src[0], expected_flat);
    ggml_cgraph* post_graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(post_graph, layout.outputs[0]);
    ggml_build_forward_expand(post_graph, layout.outputs[1]);
    ggml_build_forward_expand(post_graph, layout.outputs[2]);
    run_graph(backend, post_graph);
    check_close(tensor_get_f32(layout.outputs[0]),
                q_expected,
                "sp seq_to_head batched q output");
    check_close(tensor_get_f32(layout.outputs[1]),
                k_expected,
                "sp seq_to_head batched k output");
    check_close(tensor_get_f32(layout.outputs[2]),
                v_expected,
                "sp seq_to_head batched v output");

    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
}

void run_seq_to_head_batched_mixed_layout_test(edgedit::parallel::ProcessGroup& group,
                                               ggml_backend_t backend) {
    const int rank = group.rank();
    const int world_size = group.size();
    if (world_size != 2) {
        throw std::runtime_error("run_seq_to_head_batched_mixed_layout_test expects world_size == 2");
    }

    ggml_context* ctx = new_test_context();

    const int hidden = 2;
    const int heads = 4;
    const int shard_heads = heads / world_size;
    const int shard_seq = 3;
    ggml_tensor* q = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, hidden, heads, shard_seq, 1);
    ggml_tensor* k = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, hidden, heads, shard_seq, 1);
    ggml_tensor* v = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, hidden, heads, shard_seq, 1);
    ggml_set_name(q, "sp_seq_to_head_batched_mixed_q");
    ggml_set_name(k, "sp_seq_to_head_batched_mixed_k");
    ggml_set_name(v, "sp_seq_to_head_batched_mixed_v");

    sd::ggml_graph_cut::clear_comm_marks();
    edgedit::parallel::SPAllToAll4DBatchLayout layout =
        edgedit::parallel::sp_all_to_all_4d_seq_to_head_batched_mixed(ctx,
                                                                      {q, k, v},
                                                                      {true, true, false},
                                                                      world_size,
                                                                      "sp_test_seq_to_head_batched_mixed");

    if (layout.outputs.size() != 3 ||
        layout.outputs[0]->ne[0] != hidden ||
        layout.outputs[0]->ne[1] != shard_seq * world_size ||
        layout.outputs[0]->ne[2] != shard_heads ||
        layout.outputs[1]->ne[0] != hidden ||
        layout.outputs[1]->ne[1] != shard_seq * world_size ||
        layout.outputs[1]->ne[2] != shard_heads ||
        layout.outputs[2]->ne[0] != hidden ||
        layout.outputs[2]->ne[1] != shard_heads ||
        layout.outputs[2]->ne[2] != shard_seq * world_size ||
        layout.count_per_peer != static_cast<size_t>(hidden * 3 * shard_heads * shard_seq)) {
        throw std::runtime_error("sp_all_to_all_4d_seq_to_head_batched_mixed returned wrong metadata or shape");
    }

    ggml_cgraph* pre_graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(pre_graph, layout.send_flat);
    ggml_build_forward_expand(pre_graph, layout.recv_flat);

    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (buffer == nullptr) {
        ggml_free(ctx);
        throw std::runtime_error("alloc ctx tensors failed in seq_to_head batched mixed layout test");
    }

    std::vector<float> q_values = make_seq_to_head_input(rank, hidden, heads, shard_seq);
    std::vector<float> k_values = q_values;
    std::vector<float> v_values = q_values;
    for (float& value : k_values) {
        value += 10000.0f;
    }
    for (float& value : v_values) {
        value += 20000.0f;
    }
    tensor_set_f32(q, q_values);
    tensor_set_f32(k, k_values);
    tensor_set_f32(v, v_values);
    tensor_set_f32(layout.recv_flat->src[0],
                   std::vector<float>(static_cast<size_t>(ggml_nelements(layout.recv_flat)), 0.0f));

    run_graph(backend, pre_graph);

    const int send_idx = find_node_index(pre_graph, layout.send_flat, "seq_to_head_batched_mixed.send_flat");
    const int recv_idx = find_node_index(pre_graph, layout.recv_flat, "seq_to_head_batched_mixed.recv_flat");
    sd::ggml_graph_cut::Plan base_plan =
        make_single_segment_plan(pre_graph, {send_idx, recv_idx}, {recv_idx});
    sd::ggml_graph_cut::Segment segment =
        attach_one_comm_or_throw(pre_graph, base_plan, "sp_seq_to_head_batched_mixed");

    if (segment.comm_ops.front().kind != sd::ggml_graph_cut::Segment::CommKind::ALL_TO_ALL ||
        segment.comm_ops.front().count_per_peer != layout.count_per_peer) {
        throw std::runtime_error("sp_seq_to_head_batched_mixed expected one ALL_TO_ALL comm op with layout count_per_peer");
    }

    execute_segment_comm_or_throw(group, pre_graph, segment, "sp_seq_to_head_batched_mixed");

    std::vector<float> q_expected =
        make_seq_to_head_expected(rank, world_size, hidden, shard_heads, shard_seq);
    std::vector<float> k_expected = q_expected;
    std::vector<float> v_expected = q_expected;
    for (float& value : k_expected) {
        value += 10000.0f;
    }
    for (float& value : v_expected) {
        value += 20000.0f;
    }

    std::vector<float> expected_flat;
    expected_flat.reserve(q_expected.size() + k_expected.size() + v_expected.size());
    for (int src = 0; src < world_size; ++src) {
        for (int seq = 0; seq < shard_seq; ++seq) {
            for (int local_head = 0; local_head < shard_heads; ++local_head) {
                const int full_head = rank * shard_heads + local_head;
                for (int h = 0; h < hidden; ++h) {
                    expected_flat.push_back(a2a_seq_to_head_value(src, seq, full_head, h));
                }
                for (int h = 0; h < hidden; ++h) {
                    expected_flat.push_back(a2a_seq_to_head_value(src, seq, full_head, h) + 10000.0f);
                }
                for (int h = 0; h < hidden; ++h) {
                    expected_flat.push_back(a2a_seq_to_head_value(src, seq, full_head, h) + 20000.0f);
                }
            }
        }
    }
    check_close(tensor_get_f32(layout.recv_flat),
                expected_flat,
                "sp seq_to_head batched mixed flat recv");

    tensor_set_f32(layout.recv_flat->src[0], expected_flat);
    ggml_cgraph* post_graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(post_graph, layout.outputs[0]);
    ggml_build_forward_expand(post_graph, layout.outputs[1]);
    ggml_build_forward_expand(post_graph, layout.outputs[2]);
    run_graph(backend, post_graph);

    std::vector<float> q_seq_major_expected =
        make_seq_to_head_seq_major_expected(rank, world_size, hidden, shard_heads, shard_seq);
    std::vector<float> k_seq_major_expected = q_seq_major_expected;
    for (float& value : k_seq_major_expected) {
        value += 10000.0f;
    }

    check_close(tensor_get_f32(layout.outputs[0]),
                q_seq_major_expected,
                "sp seq_to_head batched mixed q seq-major output");
    check_close(tensor_get_f32(layout.outputs[1]),
                k_seq_major_expected,
                "sp seq_to_head batched mixed k seq-major output");
    check_close(tensor_get_f32(layout.outputs[2]),
                v_expected,
                "sp seq_to_head batched mixed v output");

    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
}

void run_seq_to_head_batched_packed_view_input_test(edgedit::parallel::ProcessGroup& group,
                                                   ggml_backend_t backend) {
    const int rank = group.rank();
    const int world_size = group.size();
    if (world_size != 2) {
        throw std::runtime_error("run_seq_to_head_batched_packed_view_input_test expects world_size == 2");
    }

    ggml_context* ctx = new_test_context();

    const int hidden = 2;
    const int heads = 4;
    const int shard_heads = heads / world_size;
    const int shard_seq = 3;
    ggml_tensor* packed = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, hidden * heads * 3, shard_seq, 1, 1);
    ggml_set_name(packed, "sp_seq_to_head_batched_packed_qkv");

    ggml_tensor* q = ggml_view_4d(ctx,
                                  packed,
                                  hidden,
                                  heads,
                                  shard_seq,
                                  1,
                                  packed->nb[0] * hidden,
                                  packed->nb[1],
                                  packed->nb[2],
                                  0);
    ggml_tensor* k = ggml_view_4d(ctx,
                                  packed,
                                  hidden,
                                  heads,
                                  shard_seq,
                                  1,
                                  packed->nb[0] * hidden,
                                  packed->nb[1],
                                  packed->nb[2],
                                  packed->nb[0] * hidden * heads);
    ggml_tensor* v = ggml_view_4d(ctx,
                                  packed,
                                  hidden,
                                  heads,
                                  shard_seq,
                                  1,
                                  packed->nb[0] * hidden,
                                  packed->nb[1],
                                  packed->nb[2],
                                  packed->nb[0] * hidden * heads * 2);
    ggml_set_name(q, "sp_seq_to_head_batched_packed_q");
    ggml_set_name(k, "sp_seq_to_head_batched_packed_k");
    ggml_set_name(v, "sp_seq_to_head_batched_packed_v");

    sd::ggml_graph_cut::clear_comm_marks();
    edgedit::parallel::SPAllToAll4DBatchLayout layout =
        edgedit::parallel::sp_all_to_all_4d_seq_to_head_batched(ctx,
                                                                {q, k, v},
                                                                world_size,
                                                                "sp_test_seq_to_head_batched_packed");

    if (layout.outputs.size() != 3 ||
        layout.outputs[0]->ne[0] != hidden ||
        layout.outputs[0]->ne[1] != shard_heads ||
        layout.outputs[0]->ne[2] != shard_seq * world_size ||
        layout.count_per_peer != static_cast<size_t>(hidden * 3 * shard_heads * shard_seq)) {
        throw std::runtime_error("sp_all_to_all_4d_seq_to_head_batched packed view returned wrong metadata or shape");
    }

    ggml_cgraph* pre_graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(pre_graph, layout.send_flat);
    ggml_build_forward_expand(pre_graph, layout.recv_flat);

    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (buffer == nullptr) {
        ggml_free(ctx);
        throw std::runtime_error("alloc ctx tensors failed in seq_to_head batched packed view test");
    }

    std::vector<float> packed_values;
    packed_values.reserve(static_cast<size_t>(hidden * heads * 3 * shard_seq));
    for (int seq = 0; seq < shard_seq; ++seq) {
        for (int part = 0; part < 3; ++part) {
            for (int head = 0; head < heads; ++head) {
                const float offset = static_cast<float>(part * 10000);
                for (int h = 0; h < hidden; ++h) {
                    packed_values.push_back(a2a_seq_to_head_value(rank, seq, head, h) + offset);
                }
            }
        }
    }
    tensor_set_f32(packed, packed_values);
    tensor_set_f32(layout.recv_flat->src[0],
                   std::vector<float>(static_cast<size_t>(ggml_nelements(layout.recv_flat)), 0.0f));

    run_graph(backend, pre_graph);

    const int send_idx = find_node_index(pre_graph, layout.send_flat, "seq_to_head_batched_packed.send_flat");
    const int recv_idx = find_node_index(pre_graph, layout.recv_flat, "seq_to_head_batched_packed.recv_flat");
    sd::ggml_graph_cut::Plan base_plan =
        make_single_segment_plan(pre_graph, {send_idx, recv_idx}, {recv_idx});
    sd::ggml_graph_cut::Segment segment =
        attach_one_comm_or_throw(pre_graph, base_plan, "sp_seq_to_head_batched_packed");

    if (segment.comm_ops.front().kind != sd::ggml_graph_cut::Segment::CommKind::ALL_TO_ALL ||
        segment.comm_ops.front().count_per_peer != layout.count_per_peer) {
        throw std::runtime_error("sp_seq_to_head_batched_packed expected one ALL_TO_ALL comm op with layout count_per_peer");
    }

    execute_segment_comm_or_throw(group, pre_graph, segment, "sp_seq_to_head_batched_packed");

    std::vector<float> q_expected =
        make_seq_to_head_expected(rank, world_size, hidden, shard_heads, shard_seq);
    std::vector<float> k_expected = q_expected;
    std::vector<float> v_expected = q_expected;
    for (float& value : k_expected) {
        value += 10000.0f;
    }
    for (float& value : v_expected) {
        value += 20000.0f;
    }

    std::vector<float> expected_flat;
    expected_flat.reserve(q_expected.size() + k_expected.size() + v_expected.size());
    for (int src = 0; src < world_size; ++src) {
        for (int seq = 0; seq < shard_seq; ++seq) {
            for (int local_head = 0; local_head < shard_heads; ++local_head) {
                const int full_head = rank * shard_heads + local_head;
                for (int h = 0; h < hidden; ++h) {
                    expected_flat.push_back(a2a_seq_to_head_value(src, seq, full_head, h));
                }
                for (int h = 0; h < hidden; ++h) {
                    expected_flat.push_back(a2a_seq_to_head_value(src, seq, full_head, h) + 10000.0f);
                }
                for (int h = 0; h < hidden; ++h) {
                    expected_flat.push_back(a2a_seq_to_head_value(src, seq, full_head, h) + 20000.0f);
                }
            }
        }
    }
    check_close(tensor_get_f32(layout.recv_flat),
                expected_flat,
                "sp seq_to_head batched packed flat recv");

    tensor_set_f32(layout.recv_flat->src[0], expected_flat);
    ggml_cgraph* post_graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(post_graph, layout.outputs[0]);
    ggml_build_forward_expand(post_graph, layout.outputs[1]);
    ggml_build_forward_expand(post_graph, layout.outputs[2]);
    run_graph(backend, post_graph);
    check_close(tensor_get_f32(layout.outputs[0]),
                q_expected,
                "sp seq_to_head batched packed q output");
    check_close(tensor_get_f32(layout.outputs[1]),
                k_expected,
                "sp seq_to_head batched packed k output");
    check_close(tensor_get_f32(layout.outputs[2]),
                v_expected,
                "sp seq_to_head batched packed v output");

    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
}

std::vector<float> make_head_to_seq_input(int rank,
                                          int hidden,
                                          int shard_heads,
                                          int sequence) {
    std::vector<float> values;
    values.reserve(static_cast<size_t>(hidden * shard_heads * sequence));
    for (int seq = 0; seq < sequence; ++seq) {
        for (int local_head = 0; local_head < shard_heads; ++local_head) {
            for (int h = 0; h < hidden; ++h) {
                values.push_back(a2a_head_to_seq_value(rank, seq, local_head, h));
            }
        }
    }
    return values;
}

std::vector<float> make_head_to_seq_flat_expected(int rank,
                                                  int world_size,
                                                  int hidden,
                                                  int shard_heads,
                                                  int shard_seq) {
    std::vector<float> expected;
    expected.reserve(static_cast<size_t>(hidden * shard_heads * shard_seq * world_size));
    for (int src = 0; src < world_size; ++src) {
        for (int seq = 0; seq < shard_seq; ++seq) {
            const int global_seq = rank * shard_seq + seq;
            for (int local_head = 0; local_head < shard_heads; ++local_head) {
                for (int h = 0; h < hidden; ++h) {
                    expected.push_back(a2a_head_to_seq_value(src, global_seq, local_head, h));
                }
            }
        }
    }
    return expected;
}

std::vector<float> make_head_to_seq_output_expected(int rank,
                                                    int world_size,
                                                    int hidden,
                                                    int shard_heads,
                                                    int shard_seq) {
    std::vector<float> expected;
    expected.reserve(static_cast<size_t>(hidden * shard_heads * world_size * shard_seq));
    for (int seq = 0; seq < shard_seq; ++seq) {
        const int global_seq = rank * shard_seq + seq;
        for (int src = 0; src < world_size; ++src) {
            for (int local_head = 0; local_head < shard_heads; ++local_head) {
                for (int h = 0; h < hidden; ++h) {
                    expected.push_back(a2a_head_to_seq_value(src, global_seq, local_head, h));
                }
            }
        }
    }
    return expected;
}

void run_head_to_seq_test(edgedit::parallel::ProcessGroup& group,
                          ggml_backend_t backend) {
    const int rank = group.rank();
    const int world_size = group.size();
    if (world_size != 2) {
        throw std::runtime_error("run_head_to_seq_test expects world_size == 2");
    }

    ggml_context* ctx = new_test_context();

    const int hidden = 2;
    const int shard_heads = 2;
    const int sequence = 6;
    const int shard_seq = sequence / world_size;
    ggml_tensor* input = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, hidden, shard_heads, sequence, 1);
    ggml_set_name(input, "sp_head_to_seq_input");

    sd::ggml_graph_cut::clear_comm_marks();
    edgedit::parallel::SPAllToAll4DLayout layout =
        edgedit::parallel::sp_all_to_all_4d_head_to_seq(ctx,
                                                        input,
                                                        world_size,
                                                        "sp_test_head_to_seq");

    if (layout.output->ne[0] != hidden ||
        layout.output->ne[1] != shard_heads * world_size ||
        layout.output->ne[2] != shard_seq ||
        layout.count_per_peer != static_cast<size_t>(hidden * shard_heads * shard_seq)) {
        throw std::runtime_error("sp_all_to_all_4d_head_to_seq returned wrong metadata or shape");
    }

    ggml_cgraph* pre_graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(pre_graph, layout.send_flat);
    ggml_build_forward_expand(pre_graph, layout.recv_flat);

    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (buffer == nullptr) {
        ggml_free(ctx);
        throw std::runtime_error("alloc ctx tensors failed in head_to_seq test");
    }

    tensor_set_f32(input, make_head_to_seq_input(rank, hidden, shard_heads, sequence));
    tensor_set_f32(layout.recv_flat->src[0],
                   std::vector<float>(static_cast<size_t>(ggml_nelements(layout.recv_flat)), 0.0f));

    run_graph(backend, pre_graph);

    const int send_idx = find_node_index(pre_graph, layout.send_flat, "head_to_seq.send_flat");
    const int recv_idx = find_node_index(pre_graph, layout.recv_flat, "head_to_seq.recv_flat");
    sd::ggml_graph_cut::Plan base_plan =
        make_single_segment_plan(pre_graph, {send_idx, recv_idx}, {recv_idx});
    sd::ggml_graph_cut::Segment segment =
        attach_one_comm_or_throw(pre_graph, base_plan, "sp_head_to_seq");

    if (segment.comm_ops.front().kind != sd::ggml_graph_cut::Segment::CommKind::ALL_TO_ALL ||
        segment.comm_ops.front().count_per_peer != layout.count_per_peer) {
        throw std::runtime_error("sp_head_to_seq expected ALL_TO_ALL comm op with layout count_per_peer");
    }

    execute_segment_comm_or_throw(group, pre_graph, segment, "sp_head_to_seq");

    std::vector<float> expected_flat =
        make_head_to_seq_flat_expected(rank, world_size, hidden, shard_heads, shard_seq);
    check_close(tensor_get_f32(layout.recv_flat),
                expected_flat,
                "sp head_to_seq flat recv");

    tensor_set_f32(layout.recv_flat->src[0], expected_flat);
    ggml_cgraph* post_graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(post_graph, layout.output);
    run_graph(backend, post_graph);
    check_close(tensor_get_f32(layout.output),
                make_head_to_seq_output_expected(rank, world_size, hidden, shard_heads, shard_seq),
                "sp head_to_seq post layout");

    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
}

void run_head_to_seq_batched_test(edgedit::parallel::ProcessGroup& group,
                                  ggml_backend_t backend) {
    const int rank = group.rank();
    const int world_size = group.size();
    if (world_size != 2) {
        throw std::runtime_error("run_head_to_seq_batched_test expects world_size == 2");
    }

    ggml_context* ctx = new_test_context();

    const int hidden = 2;
    const int shard_heads = 2;
    const int txt_sequence = 6;
    const int img_sequence = 4;
    const int txt_shard_seq = txt_sequence / world_size;
    const int img_shard_seq = img_sequence / world_size;
    ggml_tensor* txt = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, hidden, shard_heads, txt_sequence, 1);
    ggml_tensor* img = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, hidden, shard_heads, img_sequence, 1);
    ggml_set_name(txt, "sp_head_to_seq_batched_txt");
    ggml_set_name(img, "sp_head_to_seq_batched_img");

    sd::ggml_graph_cut::clear_comm_marks();
    edgedit::parallel::SPAllToAll4DBatchLayout layout =
        edgedit::parallel::sp_all_to_all_4d_head_to_seq_batched(ctx,
                                                                {txt, img},
                                                                world_size,
                                                                "sp_test_head_to_seq_batched");

    if (layout.outputs.size() != 2 ||
        layout.outputs[0]->ne[0] != hidden ||
        layout.outputs[0]->ne[1] != shard_heads * world_size ||
        layout.outputs[0]->ne[2] != txt_shard_seq ||
        layout.outputs[1]->ne[0] != hidden ||
        layout.outputs[1]->ne[1] != shard_heads * world_size ||
        layout.outputs[1]->ne[2] != img_shard_seq ||
        layout.count_per_peer != static_cast<size_t>(hidden * shard_heads * (txt_shard_seq + img_shard_seq))) {
        throw std::runtime_error("sp_all_to_all_4d_head_to_seq_batched returned wrong metadata or shape");
    }

    ggml_cgraph* pre_graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(pre_graph, layout.send_flat);
    ggml_build_forward_expand(pre_graph, layout.recv_flat);

    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (buffer == nullptr) {
        ggml_free(ctx);
        throw std::runtime_error("alloc ctx tensors failed in head_to_seq batched test");
    }

    std::vector<float> txt_values = make_head_to_seq_input(rank, hidden, shard_heads, txt_sequence);
    std::vector<float> img_values = make_head_to_seq_input(rank, hidden, shard_heads, img_sequence);
    for (float& value : img_values) {
        value += 10000.0f;
    }
    tensor_set_f32(txt, txt_values);
    tensor_set_f32(img, img_values);
    tensor_set_f32(layout.recv_flat->src[0],
                   std::vector<float>(static_cast<size_t>(ggml_nelements(layout.recv_flat)), 0.0f));

    run_graph(backend, pre_graph);

    const int send_idx = find_node_index(pre_graph, layout.send_flat, "head_to_seq_batched.send_flat");
    const int recv_idx = find_node_index(pre_graph, layout.recv_flat, "head_to_seq_batched.recv_flat");
    sd::ggml_graph_cut::Plan base_plan =
        make_single_segment_plan(pre_graph, {send_idx, recv_idx}, {recv_idx});
    sd::ggml_graph_cut::Segment segment =
        attach_one_comm_or_throw(pre_graph, base_plan, "sp_head_to_seq_batched");

    if (segment.comm_ops.front().kind != sd::ggml_graph_cut::Segment::CommKind::ALL_TO_ALL ||
        segment.comm_ops.front().count_per_peer != layout.count_per_peer) {
        throw std::runtime_error("sp_head_to_seq_batched expected one ALL_TO_ALL comm op with layout count_per_peer");
    }

    execute_segment_comm_or_throw(group, pre_graph, segment, "sp_head_to_seq_batched");

    std::vector<float> expected_flat;
    expected_flat.reserve(static_cast<size_t>(hidden * shard_heads * (txt_shard_seq + img_shard_seq) * world_size));
    for (int src = 0; src < world_size; ++src) {
        for (int seq = 0; seq < txt_shard_seq; ++seq) {
            const int global_seq = rank * txt_shard_seq + seq;
            for (int local_head = 0; local_head < shard_heads; ++local_head) {
                for (int h = 0; h < hidden; ++h) {
                    expected_flat.push_back(a2a_head_to_seq_value(src, global_seq, local_head, h));
                }
            }
        }
        for (int seq = 0; seq < img_shard_seq; ++seq) {
            const int global_seq = rank * img_shard_seq + seq;
            for (int local_head = 0; local_head < shard_heads; ++local_head) {
                for (int h = 0; h < hidden; ++h) {
                    expected_flat.push_back(a2a_head_to_seq_value(src, global_seq, local_head, h) + 10000.0f);
                }
            }
        }
    }
    check_close(tensor_get_f32(layout.recv_flat),
                expected_flat,
                "sp head_to_seq batched flat recv");

    tensor_set_f32(layout.recv_flat->src[0], expected_flat);
    ggml_cgraph* post_graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(post_graph, layout.outputs[0]);
    ggml_build_forward_expand(post_graph, layout.outputs[1]);
    run_graph(backend, post_graph);

    std::vector<float> txt_expected =
        make_head_to_seq_output_expected(rank, world_size, hidden, shard_heads, txt_shard_seq);
    std::vector<float> img_expected =
        make_head_to_seq_output_expected(rank, world_size, hidden, shard_heads, img_shard_seq);
    for (float& value : img_expected) {
        value += 10000.0f;
    }

    check_close(tensor_get_f32(layout.outputs[0]),
                txt_expected,
                "sp head_to_seq batched txt output");
    check_close(tensor_get_f32(layout.outputs[1]),
                img_expected,
                "sp head_to_seq batched img output");

    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
}

void run_roundtrip_plan_shape_test(edgedit::parallel::ProcessGroup& group,
                                   ggml_backend_t backend) {
    const int world_size = group.size();
    if (world_size != 2) {
        throw std::runtime_error("run_roundtrip_plan_shape_test expects world_size == 2");
    }

    ggml_context* ctx = new_test_context();

    const int hidden = 2;
    const int heads = 4;
    const int shard_seq = 3;
    ggml_tensor* input = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, hidden, heads, shard_seq, 1);
    ggml_set_name(input, "sp_plan_roundtrip_input");

    sd::ggml_graph_cut::clear_comm_marks();
    edgedit::parallel::SPAllToAll4DLayout seq_to_head =
        edgedit::parallel::sp_all_to_all_4d_seq_to_head(ctx,
                                                        input,
                                                        world_size,
                                                        "sp_plan_roundtrip_seq_to_head");
    edgedit::parallel::SPAllToAll4DLayout head_to_seq =
        edgedit::parallel::sp_all_to_all_4d_head_to_seq(ctx,
                                                        seq_to_head.output,
                                                        world_size,
                                                        "sp_plan_roundtrip_head_to_seq");

    ggml_cgraph* gf = ggml_new_graph_custom(ctx, 128, false);
    ggml_build_forward_expand(gf, head_to_seq.output);

    sd::ggml_graph_cut::PlanCache cache;
    std::unordered_set<const ggml_tensor*> params;
    sd::ggml_graph_cut::Plan plan =
        sd::ggml_graph_cut::resolve_plan(backend,
                                         gf,
                                         &cache,
                                         0,
                                         params,
                                         "sp_roundtrip_plan_shape");
    sd::ggml_graph_cut::clear_comm_marks();

    if (!plan.valid || !plan.has_cuts || plan.segments.size() != 3) {
        std::ostringstream oss;
        oss << "SP roundtrip plan expected 3 graph-cut segments, got "
            << plan.segments.size()
            << " valid=" << plan.valid
            << " has_cuts=" << plan.has_cuts;
        throw std::runtime_error(oss.str());
    }
    if (plan.segments[0].group_name != "sp:sp_plan_roundtrip_seq_to_head" ||
        plan.segments[1].group_name != "sp:sp_plan_roundtrip_head_to_seq" ||
        plan.segments[2].group_name != "ggml_runner.final") {
        throw std::runtime_error("SP roundtrip plan produced wrong segment ordering");
    }
    if (plan.segments[0].comm_ops.size() != 1 ||
        plan.segments[0].comm_ops.front().kind != sd::ggml_graph_cut::Segment::CommKind::ALL_TO_ALL ||
        plan.segments[0].comm_ops.front().count_per_peer != seq_to_head.count_per_peer) {
        throw std::runtime_error("SP roundtrip plan seq_to_head segment has wrong comm op");
    }
    if (plan.segments[1].comm_ops.size() != 1 ||
        plan.segments[1].comm_ops.front().kind != sd::ggml_graph_cut::Segment::CommKind::ALL_TO_ALL ||
        plan.segments[1].comm_ops.front().count_per_peer != head_to_seq.count_per_peer) {
        throw std::runtime_error("SP roundtrip plan head_to_seq segment has wrong comm op");
    }
    if (!plan.segments[2].comm_ops.empty()) {
        throw std::runtime_error("SP roundtrip final segment should not have comm ops");
    }

    ggml_free(ctx);
}

class SPGraphCutRunner final : public GGMLRunner {
public:
    explicit SPGraphCutRunner(ggml_backend_t backend)
        : GGMLRunner(backend, false) {}

    std::string get_desc() override {
        return "SPGraphCutRunner";
    }

    std::optional<sd::Tensor<float>> run_split_gather_segmented_test(
        edgedit::parallel::ProcessGroup& group
    ) {
        set_process_group(make_non_owning_process_group_ref(group));

        const int rank = group.rank();
        const int world_size = group.size();
        const int hidden = 2;
        const int seq = 7;
        std::vector<float> input_values = make_sequence_full_values(hidden, seq);

        return compute<float>(
            [&]() -> ggml_cgraph* {
                ggml_tensor* full = ggml_new_tensor_4d(compute_ctx,
                                                        GGML_TYPE_F32,
                                                        hidden,
                                                        seq,
                                                        1,
                                                        1);
                ggml_set_name(full, "sp_runner_sequence_full");
                set_backend_tensor_data(full, input_values.data());

                edgedit::parallel::SPSequenceSplit split =
                    edgedit::parallel::sp_split_sequence(compute_ctx,
                                                         full,
                                                         rank,
                                                         world_size,
                                                         1,
                                                         "sp_runner_sequence_split");
                edgedit::parallel::SPSequenceGather gather =
                    edgedit::parallel::sp_mark_gather_sequence(compute_ctx,
                                                               split.local,
                                                               world_size,
                                                               1,
                                                               split.pad,
                                                               "sp_runner_gather_sequence");

                ggml_cgraph* gf = new_graph_custom(64);
                ggml_build_forward_expand(gf, gather.gathered);
                return gf;
            },
            1,
            true,
            false
        );
    }

    std::optional<sd::Tensor<float>> run_split_gather_no_pad_segmented_test(
        edgedit::parallel::ProcessGroup& group
    ) {
        set_process_group(make_non_owning_process_group_ref(group));

        const int rank = group.rank();
        const int world_size = group.size();
        const int hidden = 3;
        const int seq = 8;
        std::vector<float> input_values = make_sequence_full_values(hidden, seq);

        return compute<float>(
            [&]() -> ggml_cgraph* {
                ggml_tensor* full = ggml_new_tensor_4d(compute_ctx,
                                                        GGML_TYPE_F32,
                                                        hidden,
                                                        seq,
                                                        1,
                                                        1);
                ggml_set_name(full, "sp_runner_sequence_no_pad_full");
                set_backend_tensor_data(full, input_values.data());

                edgedit::parallel::SPSequenceSplit split =
                    edgedit::parallel::sp_split_sequence(compute_ctx,
                                                         full,
                                                         rank,
                                                         world_size,
                                                         1,
                                                         "sp_runner_sequence_no_pad_split");
                edgedit::parallel::SPSequenceGather gather =
                    edgedit::parallel::sp_mark_gather_sequence(compute_ctx,
                                                               split.local,
                                                               world_size,
                                                               1,
                                                               split.pad,
                                                               "sp_runner_gather_sequence_no_pad");

                ggml_cgraph* gf = new_graph_custom(64);
                ggml_build_forward_expand(gf, gather.gathered);
                return gf;
            },
            1,
            true,
            false
        );
    }

    std::optional<sd::Tensor<float>> run_seq_to_head_segmented_test(
        edgedit::parallel::ProcessGroup& group
    ) {
        set_process_group(make_non_owning_process_group_ref(group));

        const int rank = group.rank();
        const int world_size = group.size();
        const int hidden = 2;
        const int heads = 4;
        const int shard_seq = 3;
        std::vector<float> input_values =
            make_seq_to_head_input(rank, hidden, heads, shard_seq);

        return compute<float>(
            [&]() -> ggml_cgraph* {
                ggml_tensor* input = ggml_new_tensor_4d(compute_ctx,
                                                         GGML_TYPE_F32,
                                                         hidden,
                                                         heads,
                                                         shard_seq,
                                                         1);
                ggml_set_name(input, "sp_runner_seq_to_head_input");
                set_backend_tensor_data(input, input_values.data());

                edgedit::parallel::SPAllToAll4DLayout layout =
                    edgedit::parallel::sp_all_to_all_4d_seq_to_head(compute_ctx,
                                                                    input,
                                                                    world_size,
                                                                    "sp_runner_seq_to_head");

                ggml_cgraph* gf = new_graph_custom(64);
                ggml_build_forward_expand(gf, layout.output);
                return gf;
            },
            1,
            true,
            false
        );
    }

    std::optional<sd::Tensor<float>> run_head_to_seq_segmented_test(
        edgedit::parallel::ProcessGroup& group
    ) {
        set_process_group(make_non_owning_process_group_ref(group));

        const int rank = group.rank();
        const int hidden = 2;
        const int shard_heads = 2;
        const int sequence = 6;
        std::vector<float> input_values =
            make_head_to_seq_input(rank, hidden, shard_heads, sequence);

        return compute<float>(
            [&]() -> ggml_cgraph* {
                ggml_tensor* input = ggml_new_tensor_4d(compute_ctx,
                                                         GGML_TYPE_F32,
                                                         hidden,
                                                         shard_heads,
                                                         sequence,
                                                         1);
                ggml_set_name(input, "sp_runner_head_to_seq_input");
                set_backend_tensor_data(input, input_values.data());

                edgedit::parallel::SPAllToAll4DLayout layout =
                    edgedit::parallel::sp_all_to_all_4d_head_to_seq(compute_ctx,
                                                                    input,
                                                                    group.size(),
                                                                    "sp_runner_head_to_seq");

                ggml_cgraph* gf = new_graph_custom(64);
                ggml_build_forward_expand(gf, layout.output);
                return gf;
            },
            1,
            true,
            false
        );
    }

    std::optional<sd::Tensor<float>> run_seq_head_roundtrip_segmented_test(
        edgedit::parallel::ProcessGroup& group
    ) {
        set_process_group(make_non_owning_process_group_ref(group));

        const int rank = group.rank();
        const int world_size = group.size();
        const int hidden = 2;
        const int heads = 4;
        const int shard_seq = 3;
        std::vector<float> input_values =
            make_seq_to_head_input(rank, hidden, heads, shard_seq);

        return compute<float>(
            [&]() -> ggml_cgraph* {
                ggml_tensor* input = ggml_new_tensor_4d(compute_ctx,
                                                         GGML_TYPE_F32,
                                                         hidden,
                                                         heads,
                                                         shard_seq,
                                                         1);
                ggml_set_name(input, "sp_runner_roundtrip_input");
                set_backend_tensor_data(input, input_values.data());

                edgedit::parallel::SPAllToAll4DLayout seq_to_head =
                    edgedit::parallel::sp_all_to_all_4d_seq_to_head(
                        compute_ctx,
                        input,
                        world_size,
                        "sp_runner_roundtrip_seq_to_head");
                edgedit::parallel::SPAllToAll4DLayout head_to_seq =
                    edgedit::parallel::sp_all_to_all_4d_head_to_seq(
                        compute_ctx,
                        seq_to_head.output,
                        world_size,
                        "sp_runner_roundtrip_head_to_seq");

                ggml_cgraph* gf = new_graph_custom(128);
                ggml_build_forward_expand(gf, head_to_seq.output);
                return gf;
            },
            1,
            true,
            false
        );
    }
};

void run_runner_segmented_tests(edgedit::parallel::ProcessGroup& group,
                                ggml_backend_t backend) {
    std::cout
        << "[rank " << group.rank()
        << "] run SP GGMLRunner segmented graph tests\n";

    const int world_size = group.size();
    const int hidden = 2;

    SPGraphCutRunner split_runner(backend);
    std::optional<sd::Tensor<float>> split_output =
        split_runner.run_split_gather_segmented_test(group);
    if (!split_output.has_value()) {
        throw std::runtime_error("SP runner split/gather segmented test returned nullopt");
    }
    check_close(sd_tensor_to_vector(*split_output),
                make_sequence_full_values(hidden, 7),
                "SP runner segmented split/gather");

    SPGraphCutRunner split_no_pad_runner(backend);
    std::optional<sd::Tensor<float>> split_no_pad_output =
        split_no_pad_runner.run_split_gather_no_pad_segmented_test(group);
    if (!split_no_pad_output.has_value()) {
        throw std::runtime_error("SP runner no-pad split/gather segmented test returned nullopt");
    }
    check_close(sd_tensor_to_vector(*split_no_pad_output),
                make_sequence_full_values(3, 8),
                "SP runner segmented split/gather no-pad");

    SPGraphCutRunner a2a_runner(backend);
    std::optional<sd::Tensor<float>> seq_to_head_output =
        a2a_runner.run_seq_to_head_segmented_test(group);
    if (!seq_to_head_output.has_value()) {
        throw std::runtime_error("SP runner seq_to_head segmented test returned nullopt");
    }
    const int shard_heads = 4 / world_size;
    const int shard_seq = 3;
    check_close(sd_tensor_to_vector(*seq_to_head_output),
                make_seq_to_head_expected(group.rank(),
                                          world_size,
                                          hidden,
                                          shard_heads,
                                          shard_seq),
                "SP runner segmented seq_to_head");

    SPGraphCutRunner head_to_seq_runner(backend);
    std::optional<sd::Tensor<float>> head_to_seq_output =
        head_to_seq_runner.run_head_to_seq_segmented_test(group);
    if (!head_to_seq_output.has_value()) {
        throw std::runtime_error("SP runner head_to_seq segmented test returned nullopt");
    }
    check_close(sd_tensor_to_vector(*head_to_seq_output),
                make_head_to_seq_output_expected(group.rank(),
                                                 world_size,
                                                 hidden,
                                                 shard_heads,
                                                 shard_seq),
                "SP runner segmented head_to_seq");

    SPGraphCutRunner roundtrip_runner(backend);
    std::optional<sd::Tensor<float>> roundtrip_output =
        roundtrip_runner.run_seq_head_roundtrip_segmented_test(group);
    if (!roundtrip_output.has_value()) {
        throw std::runtime_error("SP runner seq/head roundtrip segmented test returned nullopt");
    }
    check_close(sd_tensor_to_vector(*roundtrip_output),
                make_seq_to_head_input(group.rank(), hidden, 4, shard_seq),
                "SP runner segmented seq_to_head/head_to_seq roundtrip");
}

void run_sp_parallel_tests(edgedit::parallel::ProcessGroup& group,
                           const Args& args) {
    if (group.size() != 2) {
        throw std::runtime_error("sp_parallel_tests currently expect world_size == 2");
    }

    std::cout
        << "[rank " << group.rank() << "] run SP parallel helper tests"
        << " backend=" << args.backend
        << " local_rank=" << args.local_rank
        << "\n";

    run_validation_tests(group);

    ggml_backend_t backend = init_backend(args);
    std::cout
        << "[rank " << group.rank() << "] ggml backend = "
        << ggml_backend_name(backend)
        << "\n";

    run_split_gather_test(group, backend);
    run_split_gather_no_pad_test(group, backend);
    run_split_gather_batched_no_pad_test(group, backend);
    run_seq_to_head_test(group, backend);
    run_seq_to_head_batched_test(group, backend);
    run_seq_to_head_batched_mixed_layout_test(group, backend);
    run_seq_to_head_batched_packed_view_input_test(group, backend);
    run_head_to_seq_test(group, backend);
    run_head_to_seq_batched_test(group, backend);
    run_roundtrip_plan_shape_test(group, backend);
    run_runner_segmented_tests(group, backend);

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
        config.sp_parallel_size = args.world_size;

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

        run_sp_parallel_tests(*group, args);

        group->barrier();

        if (group->rank() == 0) {
            std::cout << "sp_parallel_test passed\n";
        }

        return 0;
    } catch (const std::exception& e) {
        std::cerr << "sp_parallel_test failed: " << e.what() << "\n";
        return 1;
    }
}
