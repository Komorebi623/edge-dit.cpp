#include "parallel/sp_parallel.hpp"

#include "backend/ggml/ggml_graph_cut.h"

#include <climits>
#include <sstream>
#include <stdexcept>

namespace edgedit::parallel {
namespace {

void check_context_tensor(ggml_context* ctx,
                          ggml_tensor* tensor,
                          const char* fn) {
    if (ctx == nullptr) {
        throw std::invalid_argument(std::string(fn) + " ctx is null");
    }
    if (tensor == nullptr) {
        throw std::invalid_argument(std::string(fn) + " tensor is null");
    }
}

void check_rank_world(int rank,
                      int world_size,
                      const char* fn) {
    if (world_size <= 0) {
        throw std::invalid_argument(std::string(fn) + " world_size must be positive");
    }
    if (rank < 0 || rank >= world_size) {
        std::ostringstream oss;
        oss << fn << " rank must be in [0, world_size): rank="
            << rank << " world_size=" << world_size;
        throw std::invalid_argument(oss.str());
    }
}

void check_world_size(int world_size,
                      const char* fn) {
    if (world_size <= 0) {
        throw std::invalid_argument(std::string(fn) + " world_size must be positive");
    }
}

void check_seq_dim(int seq_dim,
                   const char* fn) {
    if (seq_dim != 1) {
        std::ostringstream oss;
        oss << fn << " currently supports seq_dim == 1 for DiT [hidden, sequence, batch, 1] tensors, got "
            << seq_dim;
        throw std::invalid_argument(oss.str());
    }
}

ggml_tensor* pad_sequence_dim(ggml_context* ctx,
                              ggml_tensor* input,
                              int seq_dim,
                              int64_t pad) {
    if (pad <= 0) {
        return input;
    }
    check_seq_dim(seq_dim, "pad_sequence_dim");
    if (pad > static_cast<int64_t>(INT32_MAX)) {
        throw std::invalid_argument("pad_sequence_dim pad is too large for ggml_pad");
    }
    return ggml_pad(ctx, input, 0, static_cast<int>(pad), 0, 0);
}

ggml_tensor* view_sequence_dim_1(ggml_context* ctx,
                                 ggml_tensor* input,
                                 int64_t start,
                                 int64_t length) {
    if (start < 0 || length <= 0 || start + length > input->ne[1]) {
        std::ostringstream oss;
        oss << "view_sequence_dim_1 invalid range: start=" << start
            << " length=" << length
            << " seq_len=" << input->ne[1];
        throw std::invalid_argument(oss.str());
    }
    return ggml_view_4d(ctx,
                        input,
                        input->ne[0],
                        length,
                        input->ne[2],
                        input->ne[3],
                        input->nb[1],
                        input->nb[2],
                        input->nb[3],
                        static_cast<size_t>(start) * input->nb[1]);
}

ggml_tensor* new_recv_placeholder(ggml_context* ctx,
                                  ggml_tensor* like,
                                  ggml_tensor* dependency,
                                  int64_t ne0,
                                  int64_t ne1,
                                  int64_t ne2,
                                  int64_t ne3,
                                  const std::string& name) {
    ggml_tensor* leaf = ggml_new_tensor_4d(ctx, like->type, ne0, ne1, ne2, ne3);
    if (!name.empty()) {
        ggml_set_name(leaf, (name + "_recv_leaf").c_str());
    }
    ggml_tensor* recv = ggml_dup(ctx, leaf);
    recv->src[1] = dependency;
    if (!name.empty()) {
        ggml_set_name(recv, (name + "_recv").c_str());
    }
    return recv;
}

std::string graph_cut_group_name(const std::string& name) {
    if (name.empty()) {
        return "sp";
    }
    return "sp:" + name;
}

void mark_all_to_all_flat(ggml_tensor* send_flat,
                          ggml_tensor* recv_flat,
                          size_t count_per_peer,
                          const std::string& name) {
    sd::ggml_graph_cut::mark_comm_op(send_flat,
                                     recv_flat,
                                     sd::ggml_graph_cut::Segment::CommKind::ALL_TO_ALL,
                                     name,
                                     ReduceOp::kSum,
                                     count_per_peer);
}

} // namespace

int64_t sp_sequence_padding(int64_t seq_len,
                            int world_size) {
    check_world_size(world_size, "sp_sequence_padding");
    if (seq_len < 0) {
        throw std::invalid_argument("sp_sequence_padding seq_len must be non-negative");
    }
    if (seq_len == 0 || world_size == 1) {
        return 0;
    }
    return (static_cast<int64_t>(world_size) - (seq_len % world_size)) % world_size;
}

ggml_tensor* sp_split_sequence_view(ggml_context* ctx,
                                    ggml_tensor* input,
                                    int rank,
                                    int world_size,
                                    int seq_dim,
                                    int64_t* pad_out) {
    check_context_tensor(ctx, input, "sp_split_sequence_view");
    check_rank_world(rank, world_size, "sp_split_sequence_view");
    check_seq_dim(seq_dim, "sp_split_sequence_view");

    const int64_t pad = sp_sequence_padding(input->ne[seq_dim], world_size);
    if (pad_out != nullptr) {
        *pad_out = pad;
    }

    ggml_tensor* padded = pad_sequence_dim(ctx, input, seq_dim, pad);
    const int64_t local_seq_len = padded->ne[seq_dim] / world_size;
    return view_sequence_dim_1(ctx,
                               padded,
                               static_cast<int64_t>(rank) * local_seq_len,
                               local_seq_len);
}

SPSequenceSplit sp_split_sequence(ggml_context* ctx,
                                  ggml_tensor* input,
                                  int rank,
                                  int world_size,
                                  int seq_dim,
                                  const std::string& name_prefix) {
    check_context_tensor(ctx, input, "sp_split_sequence");
    check_rank_world(rank, world_size, "sp_split_sequence");
    check_seq_dim(seq_dim, "sp_split_sequence");

    SPSequenceSplit split;
    split.rank = rank;
    split.world_size = world_size;
    split.seq_dim = seq_dim;
    split.original_seq_len = input->ne[seq_dim];
    split.pad = sp_sequence_padding(split.original_seq_len, world_size);
    split.padded_seq_len = split.original_seq_len + split.pad;
    split.local_seq_len = split.padded_seq_len / world_size;

    split.input_padded = pad_sequence_dim(ctx, input, seq_dim, split.pad);
    if (!name_prefix.empty() && split.input_padded != input) {
        ggml_set_name(split.input_padded, (name_prefix + "_padded").c_str());
    }

    split.local_view = view_sequence_dim_1(ctx,
                                           split.input_padded,
                                           static_cast<int64_t>(rank) * split.local_seq_len,
                                           split.local_seq_len);
    if (!name_prefix.empty()) {
        ggml_set_name(split.local_view, (name_prefix + "_local_view").c_str());
    }

    split.local = ggml_cont(ctx, split.local_view);
    if (!name_prefix.empty()) {
        ggml_set_name(split.local, (name_prefix + "_local").c_str());
    }

    return split;
}

SPSequenceGather sp_mark_gather_sequence(ggml_context* ctx,
                                         ggml_tensor* local,
                                         int world_size,
                                         int seq_dim,
                                         int64_t pad,
                                         const std::string& name) {
    check_context_tensor(ctx, local, "sp_mark_gather_sequence");
    check_world_size(world_size, "sp_mark_gather_sequence");
    check_seq_dim(seq_dim, "sp_mark_gather_sequence");
    if (pad < 0 || pad >= local->ne[seq_dim] * static_cast<int64_t>(world_size)) {
        std::ostringstream oss;
        oss << "sp_mark_gather_sequence invalid pad=" << pad
            << " local_seq_len=" << local->ne[seq_dim]
            << " world_size=" << world_size;
        throw std::invalid_argument(oss.str());
    }
    if (local->ne[2] != 1 || local->ne[3] != 1) {
        throw std::invalid_argument("sp_mark_gather_sequence phase 1 supports [hidden, sequence, 1, 1] only");
    }

    SPSequenceGather gather;
    gather.world_size = world_size;
    gather.seq_dim = seq_dim;
    gather.local_seq_len = local->ne[seq_dim];
    gather.padded_seq_len = gather.local_seq_len * world_size;
    gather.pad = pad;
    gather.original_seq_len = gather.padded_seq_len - pad;
    gather.count_per_rank = static_cast<size_t>(ggml_nelements(local));

    // With batch == 1, ProcessGroup all_gather rank-major blocks are exactly
    // contiguous sequence shards laid out as [hidden, padded_sequence, 1, 1].
    gather.recv = new_recv_placeholder(ctx,
                                       local,
                                       local,
                                       local->ne[0],
                                       gather.padded_seq_len,
                                       local->ne[2],
                                       local->ne[3],
                                       name);

    sd::ggml_graph_cut::mark_comm_op(local,
                                     gather.recv,
                                     sd::ggml_graph_cut::Segment::CommKind::ALL_GATHER,
                                     name);
    sd::ggml_graph_cut::mark_graph_cut(gather.recv,
                                       graph_cut_group_name(name),
                                       name + "_recv");

    gather.gathered_padded = gather.recv;

    if (pad > 0) {
        gather.gathered = view_sequence_dim_1(ctx,
                                              gather.gathered_padded,
                                              0,
                                              gather.original_seq_len);
    } else {
        gather.gathered = ggml_dup(ctx, gather.gathered_padded);
    }
    if (!name.empty()) {
        ggml_set_name(gather.gathered, (name + "_output").c_str());
    }

    return gather;
}

SPAllToAll4DLayout sp_all_to_all_4d_seq_to_head(ggml_context* ctx,
                                                ggml_tensor* input,
                                                int world_size,
                                                const std::string& name) {
    check_context_tensor(ctx, input, "sp_all_to_all_4d_seq_to_head");
    check_world_size(world_size, "sp_all_to_all_4d_seq_to_head");
    if (input->ne[3] != 1) {
        throw std::invalid_argument("sp_all_to_all_4d_seq_to_head phase 1 supports batch == 1 only");
    }
    if (input->ne[1] % world_size != 0) {
        std::ostringstream oss;
        oss << "sp_all_to_all_4d_seq_to_head heads must be divisible by world_size: heads="
            << input->ne[1] << " world_size=" << world_size;
        throw std::invalid_argument(oss.str());
    }

    SPAllToAll4DLayout layout;
    layout.direction = SPAllToAll4DDirection::kSeqToHead;
    layout.world_size = world_size;
    layout.batch = input->ne[3];
    layout.head_dim = input->ne[0];
    layout.heads = input->ne[1];
    layout.shard_heads = input->ne[1] / world_size;
    layout.shard_sequence = input->ne[2];
    layout.sequence = input->ne[2] * world_size;

    // input [hs, hc, shard_seq, 1]
    // -> [hs, shard_heads, P, shard_seq]
    // -> [hs, shard_heads, shard_seq, P] peer-major after contiguous flatten.
    ggml_tensor* reshaped = ggml_reshape_4d(ctx,
                                            input,
                                            layout.head_dim,
                                            layout.shard_heads,
                                            world_size,
                                            layout.shard_sequence);
    ggml_tensor* peer_last = ggml_cont(ctx, ggml_permute(ctx, reshaped, 0, 1, 3, 2));
    layout.send_flat = ggml_cont_1d(ctx, peer_last, ggml_nelements(peer_last));
    if (!name.empty()) {
        ggml_set_name(layout.send_flat, (name + "_send_flat").c_str());
    }

    layout.count_per_peer = static_cast<size_t>(ggml_nelements(layout.send_flat)) /
                            static_cast<size_t>(world_size);
    layout.recv_flat = new_recv_placeholder(ctx,
                                            layout.send_flat,
                                            layout.send_flat,
                                            ggml_nelements(layout.send_flat),
                                            1,
                                            1,
                                            1,
                                            name + "_flat");

    mark_all_to_all_flat(layout.send_flat,
                         layout.recv_flat,
                         layout.count_per_peer,
                         name);
    sd::ggml_graph_cut::mark_graph_cut(layout.recv_flat,
                                       graph_cut_group_name(name),
                                       name + "_recv_flat");

    ggml_tensor* mid = ggml_reshape_4d(ctx,
                                       layout.recv_flat,
                                       layout.head_dim,
                                       layout.shard_heads,
                                       layout.shard_sequence,
                                       world_size);
    layout.output = ggml_cont_4d(ctx,
                                 mid,
                                 layout.head_dim,
                                 layout.shard_heads,
                                 layout.sequence,
                                 layout.batch);
    if (!name.empty()) {
        ggml_set_name(layout.output, (name + "_output").c_str());
    }
    return layout;
}

SPAllToAll4DLayout sp_all_to_all_4d_head_to_seq(ggml_context* ctx,
                                                ggml_tensor* input,
                                                int world_size,
                                                const std::string& name) {
    check_context_tensor(ctx, input, "sp_all_to_all_4d_head_to_seq");
    check_world_size(world_size, "sp_all_to_all_4d_head_to_seq");
    if (input->ne[3] != 1) {
        throw std::invalid_argument("sp_all_to_all_4d_head_to_seq phase 1 supports batch == 1 only");
    }
    if (input->ne[2] % world_size != 0) {
        std::ostringstream oss;
        oss << "sp_all_to_all_4d_head_to_seq sequence must be divisible by world_size: sequence="
            << input->ne[2] << " world_size=" << world_size;
        throw std::invalid_argument(oss.str());
    }

    SPAllToAll4DLayout layout;
    layout.direction = SPAllToAll4DDirection::kHeadToSeq;
    layout.world_size = world_size;
    layout.batch = input->ne[3];
    layout.head_dim = input->ne[0];
    layout.shard_heads = input->ne[1];
    layout.heads = input->ne[1] * world_size;
    layout.sequence = input->ne[2];
    layout.shard_sequence = input->ne[2] / world_size;

    // input [hs, shard_heads, seq, 1]
    // -> [hs, shard_heads, shard_seq, P], where the last dimension is the
    // destination peer. Contiguous flatten sends one contiguous block per peer.
    ggml_tensor* reshaped = ggml_reshape_4d(ctx,
                                            input,
                                            layout.head_dim,
                                            layout.shard_heads,
                                            layout.shard_sequence,
                                            world_size);
    ggml_tensor* peer_last = ggml_cont(ctx, reshaped);
    layout.send_flat = ggml_cont_1d(ctx, peer_last, ggml_nelements(peer_last));
    if (!name.empty()) {
        ggml_set_name(layout.send_flat, (name + "_send_flat").c_str());
    }

    layout.count_per_peer = static_cast<size_t>(ggml_nelements(layout.send_flat)) /
                            static_cast<size_t>(world_size);
    layout.recv_flat = new_recv_placeholder(ctx,
                                            layout.send_flat,
                                            layout.send_flat,
                                            ggml_nelements(layout.send_flat),
                                            1,
                                            1,
                                            1,
                                            name + "_flat");

    mark_all_to_all_flat(layout.send_flat,
                         layout.recv_flat,
                         layout.count_per_peer,
                         name);
    sd::ggml_graph_cut::mark_graph_cut(layout.recv_flat,
                                       graph_cut_group_name(name),
                                       name + "_recv_flat");

    ggml_tensor* mid = ggml_reshape_4d(ctx,
                                       layout.recv_flat,
                                       layout.head_dim,
                                       layout.shard_heads,
                                       layout.shard_sequence,
                                       world_size);
    ggml_tensor* heads_before_sequence = ggml_permute(ctx, mid, 0, 1, 3, 2);
    layout.output = ggml_cont_4d(ctx,
                                 heads_before_sequence,
                                 layout.head_dim,
                                 layout.heads,
                                 layout.shard_sequence,
                                 layout.batch);
    if (!name.empty()) {
        ggml_set_name(layout.output, (name + "_output").c_str());
    }
    return layout;
}

} // namespace edgedit::parallel
