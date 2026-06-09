#ifndef __ED_PARALLEL_SP_PARALLEL_HPP__
#define __ED_PARALLEL_SP_PARALLEL_HPP__

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "ggml.h"

namespace edgedit::parallel {

// Sequence-parallel helpers for DiT graph construction.
//
// Current model-facing layout is ggml [hidden, sequence, batch, 1], which
// corresponds to logical [batch, sequence, hidden]. The first implementation
// intentionally supports seq_dim == 1 for split/gather.

struct SPSequenceSplit {
    ggml_tensor* input_padded = nullptr;
    ggml_tensor* local_view   = nullptr;
    ggml_tensor* local        = nullptr;

    int rank       = 0;
    int world_size = 1;
    int seq_dim    = 1;

    int64_t original_seq_len = 0;
    int64_t padded_seq_len   = 0;
    int64_t local_seq_len    = 0;
    int64_t pad              = 0;
};

struct SPSequenceGather {
    // Graph-cut ALL_GATHER receive tensor.
    // Phase 1 supports [hidden, sequence, 1, 1], so rank-major all_gather
    // memory is already the desired full sequence layout:
    //   [hidden, padded_sequence, batch, 1]
    ggml_tensor* recv = nullptr;

    // Post-communication layout tensors:
    //   gathered_padded: [hidden, padded_sequence, batch, 1]
    //   gathered:        [hidden, original_sequence, batch, 1]
    ggml_tensor* gathered_padded = nullptr;
    ggml_tensor* gathered        = nullptr;

    int world_size = 1;
    int seq_dim    = 1;

    int64_t local_seq_len    = 0;
    int64_t padded_seq_len   = 0;
    int64_t original_seq_len = 0;
    int64_t pad              = 0;

    size_t count_per_rank = 0;
};

struct SPSequenceGatherBatch {
    ggml_tensor* send_flat = nullptr;
    ggml_tensor* recv_flat = nullptr;

    std::vector<ggml_tensor*> gathered_padded;
    std::vector<ggml_tensor*> gathered;

    int world_size = 1;
    int seq_dim    = 1;

    std::vector<int64_t> local_seq_lens;
    std::vector<int64_t> padded_seq_lens;
    std::vector<int64_t> original_seq_lens;
    std::vector<int64_t> pads;

    std::vector<size_t> counts_per_input;
    size_t count_per_rank = 0;
};

enum class SPAllToAll4DDirection {
    kSeqToHead,
    kHeadToSeq,
};

struct SPAllToAll4DLayout {
    SPAllToAll4DDirection direction = SPAllToAll4DDirection::kSeqToHead;

    // Flat graph-cut ALL_TO_ALL edge:
    //   send_flat -> recv_flat, count_per_peer elements per peer.
    ggml_tensor* send_flat = nullptr;
    ggml_tensor* recv_flat = nullptr;

    // Post-communication layout output.
    // kSeqToHead: [head_dim, shard_heads, sequence, batch]
    // kHeadToSeq: [head_dim, heads, shard_sequence, batch]
    ggml_tensor* output = nullptr;

    int world_size = 1;

    int64_t batch          = 1;
    int64_t head_dim       = 0;
    int64_t heads          = 0;
    int64_t shard_heads    = 0;
    int64_t sequence       = 0;
    int64_t shard_sequence = 0;

    size_t count_per_peer = 0;
};

struct SPAllToAll4DBatchLayout {
    SPAllToAll4DDirection direction = SPAllToAll4DDirection::kSeqToHead;

    ggml_tensor* send_flat = nullptr;
    ggml_tensor* recv_flat = nullptr;
    std::vector<ggml_tensor*> outputs;

    int world_size = 1;

    int64_t batch          = 1;
    int64_t head_dim       = 0;
    int64_t heads          = 0;
    int64_t shard_heads    = 0;
    int64_t sequence       = 0;
    int64_t shard_sequence = 0;
    int64_t total_head_dim = 0;

    std::vector<int64_t> head_dims;
    std::vector<int64_t> sequences;
    std::vector<int64_t> shard_sequences;

    size_t count_per_peer = 0;
};

int64_t sp_sequence_padding(int64_t seq_len,
                            int world_size);

// Convenience view helper matching the high-level split_sequence() idea.
// For communication or model compute, prefer sp_split_sequence(), whose
// `local` tensor is contiguous.
ggml_tensor* sp_split_sequence_view(ggml_context* ctx,
                                    ggml_tensor* input,
                                    int rank,
                                    int world_size,
                                    int seq_dim,
                                    int64_t* pad_out = nullptr);

SPSequenceSplit sp_split_sequence(ggml_context* ctx,
                                  ggml_tensor* input,
                                  int rank,
                                  int world_size,
                                  int seq_dim = 1,
                                  const std::string& name_prefix = "sp_split_sequence");

// Marks a graph-cut ALL_GATHER from a local sequence shard and builds the
// post-communication layout that restores [hidden, sequence, batch, 1].
//
// The returned recv tensor is the communication output. Build it into the
// pre-communication graph segment. Build `gathered` in the next graph segment
// after the graph-cut communication has run.
SPSequenceGather sp_mark_gather_sequence(ggml_context* ctx,
                                         ggml_tensor* local,
                                         int world_size,
                                         int seq_dim,
                                         int64_t pad,
                                         const std::string& name = "sp_gather_sequence");

SPSequenceGatherBatch sp_mark_gather_sequence_batched(ggml_context* ctx,
                                                      const std::vector<ggml_tensor*>& locals,
                                                      int world_size,
                                                      int seq_dim,
                                                      const std::vector<int64_t>& pads,
                                                      const std::string& name = "sp_gather_sequence_batched");

// 4D all-to-all layout helpers for attention tensors in ggml BSND layout:
//   [head_dim, heads, sequence, batch].
//
// The helpers build:
//   pre-layout send_flat
//   graph-cut ALL_TO_ALL mark to recv_flat
//   post-layout output
//
// Phase 1 supports batch == 1. This matches current DiT inference paths and
// avoids introducing a custom 5D layout op before the graph-cut plumbing is
// exercised.
SPAllToAll4DLayout sp_all_to_all_4d_seq_to_head(ggml_context* ctx,
                                                ggml_tensor* input,
                                                int world_size,
                                                const std::string& name = "sp_all_to_all_4d_seq_to_head");

SPAllToAll4DBatchLayout sp_all_to_all_4d_seq_to_head_batched(ggml_context* ctx,
                                                             const std::vector<ggml_tensor*>& inputs,
                                                             int world_size,
                                                             const std::string& name = "sp_all_to_all_4d_seq_to_head_batched");

enum class SPSeqToHeadOutputLayout {
    HeadMajor,
    SeqMajor,
    SeqMajorRopeInterleaved,
};

SPAllToAll4DBatchLayout sp_all_to_all_4d_seq_to_head_batched_layouts(ggml_context* ctx,
                                                                     const std::vector<ggml_tensor*>& inputs,
                                                                     const std::vector<SPSeqToHeadOutputLayout>& output_layouts,
                                                                     int world_size,
                                                                     const std::string& name = "sp_all_to_all_4d_seq_to_head_batched");

SPAllToAll4DBatchLayout sp_all_to_all_4d_seq_to_head_packed_recv_only(ggml_context* ctx,
                                                                      ggml_tensor* send_flat,
                                                                      int64_t total_head_dim,
                                                                      int64_t heads,
                                                                      int64_t shard_sequence,
                                                                      int64_t batch,
                                                                      int world_size,
                                                                      const std::string& name = "sp_all_to_all_4d_seq_to_head_packed");

// Same communication/layout contract as sp_all_to_all_4d_seq_to_head_batched,
// but selected outputs can be materialized as [head_dim, sequence, shard_heads, batch].
// This is useful when the immediate consumer wants sequence-major q/k tensors.
SPAllToAll4DBatchLayout sp_all_to_all_4d_seq_to_head_batched_mixed(ggml_context* ctx,
                                                                   const std::vector<ggml_tensor*>& inputs,
                                                                   const std::vector<bool>& output_seq_major,
                                                                   int world_size,
                                                                   const std::string& name = "sp_all_to_all_4d_seq_to_head_batched");

SPAllToAll4DLayout sp_all_to_all_4d_head_to_seq(ggml_context* ctx,
                                                ggml_tensor* input,
                                                int world_size,
                                                const std::string& name = "sp_all_to_all_4d_head_to_seq");

SPAllToAll4DBatchLayout sp_all_to_all_4d_head_to_seq_batched(ggml_context* ctx,
                                                             const std::vector<ggml_tensor*>& inputs,
                                                             int world_size,
                                                             const std::string& name = "sp_all_to_all_4d_head_to_seq_batched");

SPAllToAll4DBatchLayout sp_all_to_all_4d_head_to_seq_packed_recv_only(ggml_context* ctx,
                                                                      ggml_tensor* send_flat,
                                                                      int64_t head_dim,
                                                                      int64_t shard_heads,
                                                                      const std::vector<int64_t>& sequences,
                                                                      int world_size,
                                                                      const std::string& name = "sp_all_to_all_4d_head_to_seq_packed");

} // namespace edgedit::parallel

#endif // __ED_PARALLEL_SP_PARALLEL_HPP__
