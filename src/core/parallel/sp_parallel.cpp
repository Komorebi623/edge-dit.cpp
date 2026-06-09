#include "parallel/sp_parallel.hpp"

#include "backend/ggml/ggml_graph_cut.h"

#include <climits>
#include <sstream>
#include <stdexcept>
#include <vector>

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

ggml_tensor* flatten_for_comm_1d(ggml_context* ctx,
                                 ggml_tensor* tensor) {
    if (ggml_is_contiguous(tensor)) {
        return ggml_reshape_1d(ctx, tensor, ggml_nelements(tensor));
    }
    return ggml_cont_1d(ctx, tensor, ggml_nelements(tensor));
}

ggml_tensor* cont_4d_if_needed(ggml_context* ctx,
                               ggml_tensor* tensor,
                               int64_t ne0,
                               int64_t ne1,
                               int64_t ne2,
                               int64_t ne3) {
    if (ggml_is_contiguous(tensor)) {
        return ggml_reshape_4d(ctx, tensor, ne0, ne1, ne2, ne3);
    }
    return ggml_cont_4d(ctx, tensor, ne0, ne1, ne2, ne3);
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

SPSequenceGatherBatch sp_mark_gather_sequence_batched(ggml_context* ctx,
                                                      const std::vector<ggml_tensor*>& locals,
                                                      int world_size,
                                                      int seq_dim,
                                                      const std::vector<int64_t>& pads,
                                                      const std::string& name) {
    check_world_size(world_size, "sp_mark_gather_sequence_batched");
    check_seq_dim(seq_dim, "sp_mark_gather_sequence_batched");
    if (locals.empty()) {
        throw std::invalid_argument("sp_mark_gather_sequence_batched locals must not be empty");
    }
    if (locals.size() != pads.size()) {
        throw std::invalid_argument("sp_mark_gather_sequence_batched locals and pads size mismatch");
    }

    SPSequenceGatherBatch gather;
    gather.world_size = world_size;
    gather.seq_dim = seq_dim;
    gather.gathered_padded.reserve(locals.size());
    gather.gathered.reserve(locals.size());
    gather.local_seq_lens.reserve(locals.size());
    gather.padded_seq_lens.reserve(locals.size());
    gather.original_seq_lens.reserve(locals.size());
    gather.pads.reserve(locals.size());
    gather.counts_per_input.reserve(locals.size());

    std::vector<ggml_tensor*> send_chunks;
    send_chunks.reserve(locals.size());
    for (size_t i = 0; i < locals.size(); ++i) {
        ggml_tensor* local = locals[i];
        check_context_tensor(ctx, local, "sp_mark_gather_sequence_batched");
        if (local->ne[2] != 1 || local->ne[3] != 1) {
            throw std::invalid_argument("sp_mark_gather_sequence_batched phase 1 supports [hidden, sequence, 1, 1] only");
        }

        const int64_t pad = pads[i];
        if (pad < 0 || pad >= local->ne[seq_dim] * static_cast<int64_t>(world_size)) {
            std::ostringstream oss;
            oss << "sp_mark_gather_sequence_batched invalid pad=" << pad
                << " local_seq_len=" << local->ne[seq_dim]
                << " world_size=" << world_size;
            throw std::invalid_argument(oss.str());
        }

        const int64_t local_seq_len = local->ne[seq_dim];
        const int64_t padded_seq_len = local_seq_len * world_size;
        const int64_t original_seq_len = padded_seq_len - pad;
        const size_t count = static_cast<size_t>(ggml_nelements(local));

        gather.local_seq_lens.push_back(local_seq_len);
        gather.padded_seq_lens.push_back(padded_seq_len);
        gather.original_seq_lens.push_back(original_seq_len);
        gather.pads.push_back(pad);
        gather.counts_per_input.push_back(count);
        gather.count_per_rank += count;

        ggml_tensor* flat = flatten_for_comm_1d(ctx, local);
        if (!name.empty()) {
            ggml_set_name(flat, (name + "_send_" + std::to_string(i)).c_str());
        }
        send_chunks.push_back(flat);
    }

    gather.send_flat = send_chunks.front();
    for (size_t i = 1; i < send_chunks.size(); ++i) {
        gather.send_flat = ggml_concat(ctx, gather.send_flat, send_chunks[i], 0);
    }
    gather.send_flat = flatten_for_comm_1d(ctx, gather.send_flat);
    if (!name.empty()) {
        ggml_set_name(gather.send_flat, (name + "_send_flat").c_str());
    }

    gather.recv_flat = new_recv_placeholder(ctx,
                                            gather.send_flat,
                                            gather.send_flat,
                                            ggml_nelements(gather.send_flat) * world_size,
                                            1,
                                            1,
                                            1,
                                            name + "_flat");

    sd::ggml_graph_cut::mark_comm_op(gather.send_flat,
                                     gather.recv_flat,
                                     sd::ggml_graph_cut::Segment::CommKind::ALL_GATHER,
                                     name);
    sd::ggml_graph_cut::mark_graph_cut(gather.recv_flat,
                                       graph_cut_group_name(name),
                                       name + "_recv_flat");

    size_t input_offset = 0;
    for (size_t i = 0; i < locals.size(); ++i) {
        const size_t count = gather.counts_per_input[i];
        ggml_tensor* flat_view = nullptr;
        for (int src = 0; src < world_size; ++src) {
            const size_t offset = (static_cast<size_t>(src) * gather.count_per_rank +
                                   input_offset) *
                                  gather.recv_flat->nb[0];
            ggml_tensor* src_view = ggml_view_1d(ctx,
                                                 gather.recv_flat,
                                                 count,
                                                 offset);
            if (flat_view == nullptr) {
                flat_view = src_view;
            } else {
                flat_view = ggml_concat(ctx, flat_view, src_view, 0);
            }
        }

        ggml_tensor* padded = ggml_reshape_4d(ctx,
                                              flat_view,
                                              locals[i]->ne[0],
                                              gather.padded_seq_lens[i],
                                              locals[i]->ne[2],
                                              locals[i]->ne[3]);
        if (!name.empty()) {
            ggml_set_name(padded, (name + "_padded_" + std::to_string(i)).c_str());
        }
        gather.gathered_padded.push_back(padded);

        ggml_tensor* output = nullptr;
        if (gather.pads[i] > 0) {
            output = view_sequence_dim_1(ctx,
                                         padded,
                                         0,
                                         gather.original_seq_lens[i]);
        } else {
            output = ggml_dup(ctx, padded);
        }
        if (!name.empty()) {
            ggml_set_name(output, (name + "_output_" + std::to_string(i)).c_str());
        }
        gather.gathered.push_back(output);
        input_offset += count;
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
    layout.send_flat = flatten_for_comm_1d(ctx, peer_last);
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
    layout.output = cont_4d_if_needed(ctx,
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

SPAllToAll4DBatchLayout sp_all_to_all_4d_seq_to_head_batched(ggml_context* ctx,
                                                             const std::vector<ggml_tensor*>& inputs,
                                                             int world_size,
                                                             const std::string& name) {
    return sp_all_to_all_4d_seq_to_head_batched_mixed(ctx,
                                                      inputs,
                                                      std::vector<bool>(inputs.size(), false),
                                                      world_size,
                                                      name);
}

SPAllToAll4DBatchLayout sp_all_to_all_4d_seq_to_head_batched_mixed(ggml_context* ctx,
                                                                   const std::vector<ggml_tensor*>& inputs,
                                                                   const std::vector<bool>& output_seq_major,
                                                                   int world_size,
                                                                   const std::string& name) {
    std::vector<SPSeqToHeadOutputLayout> output_layouts;
    output_layouts.reserve(output_seq_major.size());
    for (bool seq_major : output_seq_major) {
        output_layouts.push_back(seq_major ? SPSeqToHeadOutputLayout::SeqMajor :
                                             SPSeqToHeadOutputLayout::HeadMajor);
    }
    return sp_all_to_all_4d_seq_to_head_batched_layouts(ctx,
                                                        inputs,
                                                        output_layouts,
                                                        world_size,
                                                        name);
}

SPAllToAll4DBatchLayout sp_all_to_all_4d_seq_to_head_batched_layouts(ggml_context* ctx,
                                                                     const std::vector<ggml_tensor*>& inputs,
                                                                     const std::vector<SPSeqToHeadOutputLayout>& output_layouts,
                                                                     int world_size,
                                                                     const std::string& name) {
    check_world_size(world_size, "sp_all_to_all_4d_seq_to_head_batched");
    if (inputs.empty()) {
        throw std::invalid_argument("sp_all_to_all_4d_seq_to_head_batched inputs must not be empty");
    }
    if (output_layouts.size() != inputs.size()) {
        throw std::invalid_argument("sp_all_to_all_4d_seq_to_head_batched_layouts output_layouts size mismatch");
    }

    ggml_tensor* first = inputs.front();
    check_context_tensor(ctx, first, "sp_all_to_all_4d_seq_to_head_batched");
    if (first->ne[3] != 1) {
        throw std::invalid_argument("sp_all_to_all_4d_seq_to_head_batched phase 1 supports batch == 1 only");
    }
    if (first->ne[1] % world_size != 0) {
        std::ostringstream oss;
        oss << "sp_all_to_all_4d_seq_to_head_batched heads must be divisible by world_size: heads="
            << first->ne[1] << " world_size=" << world_size;
        throw std::invalid_argument(oss.str());
    }

    SPAllToAll4DBatchLayout layout;
    layout.direction = SPAllToAll4DDirection::kSeqToHead;
    layout.world_size = world_size;
    layout.batch = first->ne[3];
    layout.heads = first->ne[1];
    layout.shard_heads = first->ne[1] / world_size;
    layout.shard_sequence = first->ne[2];
    layout.sequence = first->ne[2] * world_size;
    layout.outputs.reserve(inputs.size());
    layout.head_dims.reserve(inputs.size());

    std::vector<ggml_tensor*> contiguous_inputs;
    contiguous_inputs.reserve(inputs.size());
    for (size_t i = 0; i < inputs.size(); ++i) {
        ggml_tensor* input = inputs[i];
        check_context_tensor(ctx, input, "sp_all_to_all_4d_seq_to_head_batched");
        if (input->ne[1] != layout.heads ||
            input->ne[2] != layout.shard_sequence ||
            input->ne[3] != layout.batch) {
            std::ostringstream oss;
            oss << "sp_all_to_all_4d_seq_to_head_batched input " << i
                << " shape mismatch";
            throw std::invalid_argument(oss.str());
        }
        layout.head_dims.push_back(input->ne[0]);
        layout.total_head_dim += input->ne[0];
        contiguous_inputs.push_back(ggml_cont(ctx, input));
    }

    ggml_tensor* combined = contiguous_inputs.front();
    for (size_t i = 1; i < contiguous_inputs.size(); ++i) {
        combined = ggml_concat(ctx, combined, contiguous_inputs[i], 0);
    }
    if (!name.empty()) {
        ggml_set_name(combined, (name + "_combined").c_str());
    }

    // combined [sum_head_dim, heads, shard_seq, 1]
    // -> [sum_head_dim, shard_heads, P, shard_seq]
    // -> [sum_head_dim, shard_heads, shard_seq, P] peer-major after flatten.
    ggml_tensor* reshaped = ggml_reshape_4d(ctx,
                                            combined,
                                            layout.total_head_dim,
                                            layout.shard_heads,
                                            world_size,
                                            layout.shard_sequence);
    ggml_tensor* peer_last = ggml_cont(ctx, ggml_permute(ctx, reshaped, 0, 1, 3, 2));
    layout.send_flat = flatten_for_comm_1d(ctx, peer_last);
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
                                       layout.total_head_dim,
                                       layout.shard_heads,
                                       layout.shard_sequence,
                                       world_size);

    size_t offset = 0;
    for (size_t i = 0; i < layout.head_dims.size(); ++i) {
        const int64_t head_dim = layout.head_dims[i];
        ggml_tensor* output_view = ggml_view_4d(ctx,
                                                mid,
                                                head_dim,
                                                layout.shard_heads,
                                                layout.shard_sequence,
                                                world_size,
                                                mid->nb[1],
                                                mid->nb[2],
                                                mid->nb[3],
                                                offset);
        ggml_tensor* output = nullptr;
        if (output_layouts[i] == SPSeqToHeadOutputLayout::SeqMajor) {
            ggml_tensor* seq_before_heads = ggml_permute(ctx, output_view, 0, 3, 1, 2);
            output = cont_4d_if_needed(ctx,
                                       seq_before_heads,
                                       head_dim,
                                       layout.sequence,
                                       layout.shard_heads,
                                       layout.batch);
        } else if (output_layouts[i] == SPSeqToHeadOutputLayout::SeqMajorRopeInterleaved) {
            if (head_dim % 2 != 0) {
                throw std::invalid_argument("sp_all_to_all_4d_seq_to_head_batched_layouts rope-interleaved output requires even head_dim");
            }
            if (layout.batch != 1) {
                throw std::invalid_argument("sp_all_to_all_4d_seq_to_head_batched_layouts rope-interleaved output supports batch == 1 only");
            }
            ggml_tensor* interleaved_view = ggml_view_4d(ctx,
                                                         output_view,
                                                         2,
                                                         head_dim / 2,
                                                         layout.sequence,
                                                         layout.shard_heads,
                                                         output_view->nb[0] * 2,
                                                         output_view->nb[2],
                                                         output_view->nb[1],
                                                         0);
            ggml_tensor* half_seq_heads_interleaved = ggml_permute(ctx,
                                                                   interleaved_view,
                                                                   3,
                                                                   0,
                                                                   1,
                                                                   2);
            output = cont_4d_if_needed(ctx,
                                       half_seq_heads_interleaved,
                                       head_dim / 2,
                                       layout.sequence,
                                       layout.shard_heads,
                                       2);
        } else {
            output = cont_4d_if_needed(ctx,
                                       output_view,
                                       head_dim,
                                       layout.shard_heads,
                                       layout.sequence,
                                       layout.batch);
        }
        if (!name.empty()) {
            ggml_set_name(output, (name + "_output_" + std::to_string(i)).c_str());
        }
        layout.outputs.push_back(output);
        offset += static_cast<size_t>(head_dim) * mid->nb[0];
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
    layout.send_flat = flatten_for_comm_1d(ctx, reshaped);
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
    layout.output = cont_4d_if_needed(ctx,
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

SPAllToAll4DBatchLayout sp_all_to_all_4d_head_to_seq_batched(ggml_context* ctx,
                                                             const std::vector<ggml_tensor*>& inputs,
                                                             int world_size,
                                                             const std::string& name) {
    check_world_size(world_size, "sp_all_to_all_4d_head_to_seq_batched");
    if (inputs.empty()) {
        throw std::invalid_argument("sp_all_to_all_4d_head_to_seq_batched inputs must not be empty");
    }

    ggml_tensor* first = inputs.front();
    check_context_tensor(ctx, first, "sp_all_to_all_4d_head_to_seq_batched");
    if (first->ne[3] != 1) {
        throw std::invalid_argument("sp_all_to_all_4d_head_to_seq_batched phase 1 supports batch == 1 only");
    }
    if (first->ne[2] % world_size != 0) {
        std::ostringstream oss;
        oss << "sp_all_to_all_4d_head_to_seq_batched sequence must be divisible by world_size: sequence="
            << first->ne[2] << " world_size=" << world_size;
        throw std::invalid_argument(oss.str());
    }

    SPAllToAll4DBatchLayout layout;
    layout.direction = SPAllToAll4DDirection::kHeadToSeq;
    layout.world_size = world_size;
    layout.batch = first->ne[3];
    layout.head_dim = first->ne[0];
    layout.shard_heads = first->ne[1];
    layout.heads = first->ne[1] * world_size;
    layout.outputs.reserve(inputs.size());
    layout.sequences.reserve(inputs.size());
    layout.shard_sequences.reserve(inputs.size());

    std::vector<int64_t> chunk_nelements;
    chunk_nelements.reserve(inputs.size());
    std::vector<ggml_tensor*> send_chunks;
    send_chunks.reserve(inputs.size() * static_cast<size_t>(world_size));
    for (int dst = 0; dst < world_size; ++dst) {
        for (size_t i = 0; i < inputs.size(); ++i) {
            ggml_tensor* input = inputs[i];
            check_context_tensor(ctx, input, "sp_all_to_all_4d_head_to_seq_batched");
            if (input->ne[0] != layout.head_dim ||
                input->ne[1] != layout.shard_heads ||
                input->ne[3] != layout.batch) {
                std::ostringstream oss;
                oss << "sp_all_to_all_4d_head_to_seq_batched input " << i
                    << " shape mismatch";
                throw std::invalid_argument(oss.str());
            }
            if (input->ne[2] % world_size != 0) {
                std::ostringstream oss;
                oss << "sp_all_to_all_4d_head_to_seq_batched input " << i
                    << " sequence must be divisible by world_size: sequence="
                    << input->ne[2] << " world_size=" << world_size;
                throw std::invalid_argument(oss.str());
            }

            const int64_t shard_sequence = input->ne[2] / world_size;
            if (dst == 0) {
                layout.sequences.push_back(input->ne[2]);
                layout.shard_sequences.push_back(shard_sequence);
                layout.sequence += input->ne[2];
                layout.shard_sequence += shard_sequence;
                chunk_nelements.push_back(input->ne[0] *
                                          input->ne[1] *
                                          shard_sequence *
                                          input->ne[3]);
            }

            ggml_tensor* chunk = ggml_view_4d(ctx,
                                              input,
                                              layout.head_dim,
                                              layout.shard_heads,
                                              shard_sequence,
                                              layout.batch,
                                              input->nb[1],
                                              input->nb[2],
                                              input->nb[3],
                                              static_cast<size_t>(dst) *
                                                  static_cast<size_t>(shard_sequence) *
                                                  input->nb[2]);
            chunk = flatten_for_comm_1d(ctx, chunk);
            if (!name.empty()) {
                ggml_set_name(chunk, (name + "_send_chunk_" + std::to_string(dst) + "_" + std::to_string(i)).c_str());
            }
            send_chunks.push_back(chunk);
        }
    }

    ggml_tensor* send_flat = send_chunks.front();
    for (size_t i = 1; i < send_chunks.size(); ++i) {
        send_flat = ggml_concat(ctx, send_flat, send_chunks[i], 0);
    }
    layout.send_flat = flatten_for_comm_1d(ctx, send_flat);
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

    for (size_t i = 0; i < inputs.size(); ++i) {
        const int64_t shard_sequence = layout.shard_sequences[i];
        const int64_t chunk_ne = chunk_nelements[i];
        size_t input_peer_offset = 0;
        for (size_t j = 0; j < i; ++j) {
            input_peer_offset += static_cast<size_t>(chunk_nelements[j]);
        }

        ggml_tensor* flat_view = nullptr;
        for (int src = 0; src < world_size; ++src) {
            const size_t offset = (static_cast<size_t>(src) * layout.count_per_peer +
                                   input_peer_offset) *
                                  layout.recv_flat->nb[0];
            ggml_tensor* src_view = ggml_view_1d(ctx,
                                                 layout.recv_flat,
                                                 chunk_ne,
                                                 offset);
            if (flat_view == nullptr) {
                flat_view = src_view;
            } else {
                flat_view = ggml_concat(ctx, flat_view, src_view, 0);
            }
        }

        ggml_tensor* src_peer_major = ggml_reshape_4d(ctx,
                                                      flat_view,
                                                      layout.head_dim,
                                                      layout.shard_heads,
                                                      shard_sequence,
                                                      world_size);
        ggml_tensor* heads_before_sequence = ggml_permute(ctx, src_peer_major, 0, 1, 3, 2);
        ggml_tensor* output = ggml_cont_4d(ctx,
                                           heads_before_sequence,
                                           layout.head_dim,
                                           layout.heads,
                                           shard_sequence,
                                           layout.batch);
        if (!name.empty()) {
            ggml_set_name(output, (name + "_output_" + std::to_string(i)).c_str());
        }
        layout.outputs.push_back(output);
    }

    return layout;
}

} // namespace edgedit::parallel
