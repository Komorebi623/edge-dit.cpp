#pragma once

// H3-only graph implementation, not a fused Triton/CuTe kernel or a global
// backend switch. Define MINIMAX_H3_SPARSE_REFERENCE_ONLY for a libggml-free test.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <numeric>
#include <string>
#include <vector>
#ifndef MINIMAX_H3_SPARSE_REFERENCE_ONLY
#include "core/backend/ggml/ggml_extend.hpp"
#endif

namespace MiniMaxH3 {

// Only TargetVideo is eligible for pruning. Reference includes reference video;
// Query means a protected conditioning/query segment, NOT an output-row mask.
enum class SparseTokenKind { TargetVideo, Text, Audio, Reference, Query };
struct SparseTokenSpan {
    int64_t begin = 0;
    int64_t end = 0;  // exclusive; spans must partition [0, sequence)
    SparseTokenKind kind = SparseTokenKind::Text;
};
struct SparseAttentionLayout {
    std::vector<SparseTokenSpan> spans;
};

// Conservative metadata/node budget for the current per-head/per-Q-block
// expanded graph. This is a rejection gate, NOT a proven VRAM bound; do not
// increase it to force production clips through. Batched/native execution is
// needed for large clips. Checked arithmetic avoids overflow on bad geometry.
inline bool h3_subblock_graph_budget(const SparseAttentionLayout& layout,
                                     int64_t heads, int64_t layers, int64_t capacity,
                                     std::string* reason = nullptr) {
    auto fail = [&]() {
        if (reason) *reason = "MiniMax-H3 SubBlock expanded graph exceeds conservative capacity; bounded execution required";
        return false;
    };
    if (heads <= 0 || layers <= 0 || capacity <= 8192) return fail();
    int64_t blocks = 0, spans = 0;
    for (const auto& span : layout.spans) {
        if (span.begin < 0 || span.end <= span.begin) return fail();
        ++spans;
        if (span.kind == SparseTokenKind::TargetVideo) {
            const int64_t count = (span.end - span.begin) / 64 + ((span.end - span.begin) % 64 != 0);
            if (count > capacity - blocks) return fail();
            blocks += count;
        }
    }
    if (blocks == 0 || spans > (capacity - 128) / 64) return fail();
    // Includes router/pooling/gathers, protected-span concatenation, attention
    // prep/probes, output concatenations and baseline transformer overhead.
    const int64_t block_cost = 128 + 16 * spans;
    const int64_t span_cost = 64 * spans;
    if (blocks > (capacity - span_cost) / block_cost) return fail();
    const int64_t per_head = blocks * block_cost + span_cost;
    int64_t remaining = capacity - 8192;
    if (layers > remaining / 512) return fail();
    remaining -= layers * 512;
    if (heads > remaining / layers || per_head > remaining / layers / heads) return fail();
    if (reason) reason->clear();
    return true;
}

// The runner's bind_backend_tensor_data registers a pointer, not a copy. Keep
// this owner alive until the graph has finished executing (including replays).
// One owner per graph; release it with that graph, not on every denoise step.
struct SparseAttentionGraphInputs {
    std::vector<std::unique_ptr<uint8_t[]>> buffers;
};
struct SparseAttentionConfig {
    int sparse_block_tokens = 64;
    float sparse_keep_ratio = 0.5f;  // fraction of TARGET VIDEO blocks
    int sparse_warmup_steps = 2;    // metadata only: caller owns step/layer gates
    int sparse_topk = -1;          // -1: ceil(ratio * video_blocks); 0 is invalid
    std::shared_ptr<SparseAttentionGraphInputs> graph_inputs =
        std::make_shared<SparseAttentionGraphInputs>();
};

namespace h3_sparse_detail {
constexpr int head_dim = 128;
constexpr float baseline_kv_scale = 1.0f / 128.0f;
inline float softmax_scale() { return 1.0f / std::sqrt(float(head_dim)); }
inline bool error(std::string* out, const char* message) {
    if (out) *out = message;
    return false;
}
struct Block { int64_t begin, end; };
struct Plan {
    std::vector<Block> video;
    std::vector<SparseTokenSpan> chunks;
    int topk = 0;
    int64_t protected_tokens = 0;
};
inline bool make_plan(int64_t sequence, int64_t heads, int64_t dimension,
                      float kv_scale, const SparseAttentionConfig& config,
                      const SparseAttentionLayout& layout, Plan* plan,
                      std::string* message = nullptr) {
    if (message) message->clear();
    if (!plan || sequence <= 0 || sequence > std::numeric_limits<int32_t>::max() ||
        heads <= 0 || heads > std::numeric_limits<int64_t>::max() / head_dim ||
        dimension != head_dim || !std::isfinite(kv_scale) ||
        kv_scale != baseline_kv_scale)
        return error(message, "H3 sparse requires positive self-attention lengths, head_dim=128, kv_scale=1/128");
    if (config.sparse_block_tokens != 64 || !std::isfinite(config.sparse_keep_ratio) ||
        config.sparse_keep_ratio <= 0 || config.sparse_keep_ratio > 1 ||
        config.sparse_topk < -1 || config.sparse_topk == 0 || config.sparse_warmup_steps < 0)
        return error(message, "invalid H3 SubBlock configuration (Q64/K64, four 16-token cells)");
    Plan result;
    int64_t next = 0;
    for (const auto& span : layout.spans) {
        if (span.begin != next || span.end <= span.begin || span.end > sequence)
            return error(message, "layout must explicitly partition all real tokens, in order, without gaps/overlap");
        switch (span.kind) {
            case SparseTokenKind::TargetVideo:
                for (int64_t b = span.begin; b < span.end; b += config.sparse_block_tokens) {
                    const int64_t end = std::min(span.end, b + config.sparse_block_tokens);
                    result.video.push_back({b, end});
                    result.chunks.push_back({b, end, span.kind});
                }
                break;
            case SparseTokenKind::Text:
            case SparseTokenKind::Audio:
            case SparseTokenKind::Reference:
            case SparseTokenKind::Query:
                result.protected_tokens += span.end - span.begin;
                result.chunks.push_back(span);
                break;
            default: return error(message, "unknown sparse token kind");
        }
        next = span.end;
    }
    if (next != sequence || result.video.empty())
        return error(message, "layout must cover the sequence and contain target video (no silent dense substitute)");
    const int64_t blocks = static_cast<int64_t>(result.video.size());
    result.topk = config.sparse_topk == -1
        ? static_cast<int>(std::ceil(double(config.sparse_keep_ratio) * double(blocks)))
        : config.sparse_topk;
    if (result.topk <= 0 || result.topk > blocks)
        return error(message, "topk must be in [1, video_blocks]");
    *plan = std::move(result);
    return true;
}
inline size_t q_index(int d, int64_t token, int64_t head, int64_t length) {
    return size_t(d + head_dim * (token + length * head));
}
inline size_t v_index(int d, int64_t token, int64_t head, int64_t heads) {
    return size_t(d + head_dim * (head + heads * token));
}
inline std::vector<float> means(const float* x, const Block& b, int64_t h, int64_t length) {
    const int cells = int((b.end - b.begin + 15) / 16);
    std::vector<float> out(size_t(cells * head_dim), 0.0f);
    for (int cell = 0; cell < cells; ++cell) {
        const int64_t begin = b.begin + 16 * cell, end = std::min(b.end, begin + 16);
        for (int d = 0; d < head_dim; ++d) {
            float sum = 0;
            for (int64_t t = begin; t < end; ++t) sum += x[q_index(d, t, h, length)];
            out[size_t(cell * head_dim + d)] = sum / float(end - begin);
        }
    }
    return out;
}
inline float block_score(const std::vector<float>& qm, const std::vector<float>& km) {
    std::vector<float> scores;
    for (size_t a = 0; a < qm.size(); a += head_dim) {
        for (size_t b = 0; b < km.size(); b += head_dim) {
            float dot = 0;
            for (int d = 0; d < head_dim; ++d) dot += qm[a + d] * km[b + d];
            scores.push_back(dot * softmax_scale());
        }
    }
    const float maximum = *std::max_element(scores.begin(), scores.end());
    float sum = 0;
    for (float value : scores) sum += std::exp(value - maximum);
    return maximum + std::log(sum);
}
inline bool finite_input(const float* x, size_t count) {
    return x && std::all_of(x, x + count, [](float f) { return std::isfinite(f); });
}
} // namespace h3_sparse_detail

// selected_blocks[(head * video_blocks + query_block) * topk + rank], local
// video-block IDs. Selection order is immaterial; CPU ties prefer lower IDs.
struct SparseAttentionReference {
    std::vector<float> output;  // contiguous [128 * heads, L], not Q's layout
    std::vector<int32_t> selected_blocks;
    int video_blocks = 0;
    int topk = 0;
};

// Genuine F32 accumulation + max-subtracted softmax. q/k:[128,L,H],
// v:[128,H,L]. This is the mathematical baseline, NOT emulated flash rounding:
// flash casts K*kv_scale and V*kv_scale to F16, uses softmax_scale/kv_scale,
// then multiplies the output by 1/kv_scale. Ideal F32 cancels those KV factors.
inline bool h3_subblock_attention_reference(const float* q, const float* k, const float* v,
                                            int64_t length, int64_t heads, int64_t dimension,
                                            float kv_scale, const SparseAttentionConfig& options,
                                            const SparseAttentionLayout& layout,
                                            SparseAttentionReference* result,
                                            std::string* message = nullptr) {
    using namespace h3_sparse_detail;
    Plan plan;
    if (!result) return error(message, "missing reference output");
    if (!make_plan(length, heads, dimension, kv_scale, options, layout, &plan, message)) return false;
    if (size_t(heads) > std::numeric_limits<size_t>::max() / size_t(length) / head_dim)
        return error(message, "reference tensor size overflow");
    const size_t count = size_t(heads) * size_t(length) * head_dim;
    if (!finite_input(q, count) || !finite_input(k, count) || !finite_input(v, count))
        return error(message, "reference inputs must be finite F32");
    SparseAttentionReference out;
    out.video_blocks = int(plan.video.size());
    out.topk = plan.topk;
    out.output.resize(count);
    out.selected_blocks.resize(size_t(heads) * plan.video.size() * size_t(plan.topk));
    for (int64_t h = 0; h < heads; ++h) {
        std::vector<std::vector<float>> km;
        for (const auto& b : plan.video) km.push_back(means(k, b, h, length));
        size_t qb = 0;
        for (const auto& chunk : plan.chunks) {
            std::vector<int64_t> keys;
            if (chunk.kind != SparseTokenKind::TargetVideo) {
                keys.resize(size_t(length));
                std::iota(keys.begin(), keys.end(), int64_t(0));
            } else {
                auto qm = means(q, {chunk.begin, chunk.end}, h, length);
                std::vector<float> scores;
                for (const auto& cell : km) scores.push_back(block_score(qm, cell));
                if (!finite_input(scores.data(), scores.size()))
                    return error(message, "nonfinite router score (F32 overflow)");
                std::vector<int32_t> ids(plan.video.size());
                std::iota(ids.begin(), ids.end(), int32_t(0));
                std::partial_sort(ids.begin(), ids.begin() + plan.topk, ids.end(),
                    [&](int32_t a, int32_t b) { return scores[a] == scores[b] ? a < b : scores[a] > scores[b]; });
                for (int rank = 0; rank < plan.topk; ++rank) {
                    const int32_t id = ids[size_t(rank)];
                    out.selected_blocks[(size_t(h) * plan.video.size() + qb) * size_t(plan.topk) + size_t(rank)] = id;
                    for (int64_t t = plan.video[size_t(id)].begin; t < plan.video[size_t(id)].end; ++t) keys.push_back(t);
                }
                for (const auto& span : layout.spans) {
                    if (span.kind != SparseTokenKind::TargetVideo)
                        for (int64_t t = span.begin; t < span.end; ++t) keys.push_back(t);
                }
                ++qb;
            }
            std::vector<float> logits(keys.size());
            for (int64_t t = chunk.begin; t < chunk.end; ++t) {
                for (size_t i = 0; i < keys.size(); ++i) {
                    float dot = 0;
                    for (int d = 0; d < head_dim; ++d)
                        dot += q[q_index(d, t, h, length)] * k[q_index(d, keys[i], h, length)];
                    logits[i] = dot * softmax_scale();
                }
                if (!finite_input(logits.data(), logits.size()))
                    return error(message, "nonfinite attention logits (F32 overflow)");
                const float maximum = *std::max_element(logits.begin(), logits.end());
                float denominator = 0;
                for (auto& logit : logits) { logit = std::exp(logit - maximum); denominator += logit; }
                for (int d = 0; d < head_dim; ++d) {
                    float numerator = 0;
                    for (size_t i = 0; i < keys.size(); ++i)
                        numerator += logits[i] * v[v_index(d, keys[i], h, heads)];
                    const float value = numerator / denominator;
                    if (!std::isfinite(value)) return error(message, "nonfinite F32 output");
                    out.output[size_t(d + head_dim * (h + heads * t))] = value;
                }
            }
        }
    }
    *result = std::move(out);
    return true;
}

#ifndef MINIMAX_H3_SPARSE_REFERENCE_ONLY
namespace h3_sparse_detail {
// All generated ops are checked on this backend, never implicitly scheduled to
// another backend. The DFS stops at supplied q/k/v: their producers belong to
// the main graph, not to this helper's capability contract.
inline bool supported(GGMLRunnerContext* runner, ggml_tensor* root,
                      ggml_tensor* q, ggml_tensor* k, ggml_tensor* v) {
    std::vector<ggml_tensor*> pending{root};
    std::unordered_set<ggml_tensor*> seen;
    while (!pending.empty()) {
        auto* t = pending.back(); pending.pop_back();
        if (!t || t == q || t == k || t == v || !seen.insert(t).second) continue;
        if (t->op != GGML_OP_NONE && !ggml_backend_supports_op(runner->backend, t)) return false;
        for (auto* source : t->src) if (source) pending.push_back(source);
    }
    return true;
}
template<class T>
inline ggml_tensor* constant(GGMLRunnerContext* runner, const SparseAttentionConfig& config,
                             ggml_type type, int64_t n0, int64_t n1, const std::vector<T>& values) {
    auto* t = ggml_new_tensor_2d(runner->ggml_ctx, type, n0, n1);
    const size_t bytes = values.size() * sizeof(T);
    if (t->data) std::memcpy(t->data, values.data(), bytes);
    else {
        auto data = std::unique_ptr<uint8_t[]>(new uint8_t[bytes]);
        std::memcpy(data.get(), values.data(), bytes);
        const void* pointer = data.get();
        config.graph_inputs->buffers.push_back(std::move(data));
        ggml_set_input(t);
        runner->bind_backend_tensor_data(t, pointer);
    }
    return t;
}
inline ggml_tensor* slice(ggml_context* c, ggml_tensor* x, int64_t begin, int64_t end) {
    return ggml_view_2d(c, x, x->ne[0], end - begin, x->nb[1], size_t(begin) * x->nb[1]);
}
inline ggml_tensor* append(ggml_context* c, ggml_tensor* a, ggml_tensor* b, int dim) {
    return a ? ggml_concat(c, a, b, dim) : b;
}
inline ggml_tensor* poolmean(ggml_context* c, ggml_tensor* x, const Block& block) {
    const int64_t full = (block.end - block.begin) / 16 * 16;
    ggml_tensor* result = nullptr;
    if (full) {
        auto* cells = ggml_cont(c, slice(c, x, block.begin, block.begin + full));
        result = ggml_pool_2d(c, cells, GGML_OP_POOL_AVG, 1, 16, 1, 16, 0, 0);
    }
    if (block.begin + full < block.end) {
        auto* tail = ggml_cont(c, ggml_transpose(c, slice(c, x, block.begin + full, block.end)));
        tail = ggml_cont(c, ggml_transpose(c, ggml_mean(c, tail)));
        result = append(c, result, tail, 1); // divides by REAL tail count
    }
    return ggml_cont(c, result);
}

// Keep the existing baseline helper's kv_scale, F16 prep, output compensation
// and output layout. A masked flash preflight prevents its internal dense
// fallback from receiving a flash-oriented mask with the wrong dimensions.
inline ggml_tensor* attention(GGMLRunnerContext* runner, ggml_tensor* q, ggml_tensor* k,
                               ggml_tensor* v_rows, float kv_scale, ggml_tensor* validity) {
    auto* c = runner->ggml_ctx;
    const int64_t nq = q->ne[1], nk = k->ne[1];
    q = ggml_reshape_3d(c, ggml_cont(c, q), head_dim, nq, 1);
    k = ggml_reshape_3d(c, ggml_cont(c, k), head_dim, nk, 1);
    auto* v = ggml_reshape_3d(c, ggml_cont(c, v_rows), head_dim, 1, nk);
    ggml_tensor* mask = nullptr;
    if (validity) {
        validity = ggml_cont(c, validity);
        if (runner->flash_attn_enabled) {
            mask = ggml_repeat_4d(c, ggml_reshape_2d(c, validity, 1, nk), nq, nk, 1, 1);
            auto* flash_mask = ggml_cast(c, ggml_cont(c, ggml_transpose(c, mask)), GGML_TYPE_F16);
#ifdef GGML_KQ_MASK_PAD
            if (nq % GGML_KQ_MASK_PAD)
                flash_mask = ggml_pad(c, flash_mask, 0, GGML_PAD(nq, GGML_KQ_MASK_PAD) - nq, 0, 0);
#endif
            auto* pk = ggml_cast(c, ggml_scale(c, k, kv_scale), GGML_TYPE_F16);
            auto* pv = ggml_cast(c, ggml_scale(c, ggml_reshape_3d(c, v_rows, head_dim, nk, 1), kv_scale), GGML_TYPE_F16);
            auto* probe = ggml_flash_attn_ext(c, q, pk, pv, flash_mask, softmax_scale() / kv_scale, 0, 0);
            if (!ggml_backend_supports_op(runner->backend, probe)) return nullptr;
        } else {
            mask = ggml_repeat_4d(c, ggml_reshape_2d(c, validity, nk, 1), nk, nq, 1, 1);
        }
    }
    auto* out = ggml_ext_attention_ext(c, runner->backend, q, k, v, 1, mask,
                                       true, runner->flash_attn_enabled, kv_scale,
                                       false, false, -1, -1, true);
    if (runner->flash_attn_enabled) {
        // The helper may fall back to matmul if flash is unsupported. Do not
        // lose baseline flash rounding by presenting that fallback as success.
        bool has_flash = false;
        std::vector<ggml_tensor*> pending{out};
        std::unordered_set<ggml_tensor*> seen;
        while (!pending.empty()) {
            auto* t = pending.back(); pending.pop_back();
            if (!t || t == q || t == k || t == v || !seen.insert(t).second) continue;
            has_flash |= t->op == GGML_OP_FLASH_ATTN_EXT;
            for (auto* source : t->src) if (source) pending.push_back(source);
        }
        if (!has_flash) return nullptr;
    }
    return out;
}
} // namespace h3_sparse_detail

// Minimal integration: explicitly tag a partition of the PACKED sequence;
// target video alone is sparse, every other span is protected. Batch=1 only.
// Return nullptr on unsupported contract/op; caller must surface that failure
// (or explicitly log/choose its own baseline), never report it as sparse success.
inline ggml_tensor* h3_subblock_attention(GGMLRunnerContext* runner,
                                         ggml_tensor* q, ggml_tensor* k, ggml_tensor* v,
                                         int64_t heads, float kv_scale,
                                         const SparseAttentionConfig& options,
                                         const SparseAttentionLayout& layout) {
    using namespace h3_sparse_detail;
    Plan plan;
    if (!runner || !runner->ggml_ctx || !runner->backend || !q || !k || !v ||
        !options.graph_inputs || !h3_subblock_graph_budget(layout, heads, 1, 65536) ||
        (!runner->bind_backend_tensor_data && ggml_get_no_alloc(runner->ggml_ctx)) ||
        q->ne[3] != 1 || k->ne[3] != 1 || v->ne[3] != 1 ||
        q->ne[0] != head_dim || k->ne[0] != head_dim || v->ne[0] != head_dim ||
        q->ne[2] != heads || k->ne[2] != heads || v->ne[1] != heads ||
        k->ne[1] != q->ne[1] || v->ne[2] != q->ne[1] ||
        !make_plan(q->ne[1], heads, q->ne[0], kv_scale, options, layout, &plan)) return nullptr;
    // Baseline BF16 K scaling rounds in BF16 before its F16 cast, whereas
    // get_rows produces F32. F16 V has a different prep branch in the shared
    // helper. Until those cases are explicitly matched, fail closed rather
    // than silently changing low-memory/dtype semantics. Normal H3 Q/K/V F32.
    if (q->type != GGML_TYPE_F32 || k->type != GGML_TYPE_F32 || v->type != GGML_TYPE_F32 ||
        q->nb[0] != ggml_type_size(q->type) || k->nb[0] != ggml_type_size(k->type) ||
        v->nb[0] != ggml_type_size(v->type)) return nullptr;
    auto* c = runner->ggml_ctx;
    const int64_t blocks = int64_t(plan.video.size()), width = options.sparse_block_tokens;
    std::vector<int32_t> token_ids(size_t(blocks * width));
    std::vector<float> token_mask(size_t(blocks * width), 0);
    std::vector<int64_t> cell_start(size_t(blocks + 1), 0);
    bool ragged = false;
    for (int64_t b = 0; b < blocks; ++b) {
        const auto block = plan.video[size_t(b)];
        cell_start[size_t(b + 1)] = cell_start[size_t(b)] + (block.end - block.begin + 15) / 16;
        for (int64_t t = 0; t < width; ++t) {
            const bool real = block.begin + t < block.end;
            token_ids[size_t(b * width + t)] = int32_t(real ? block.begin + t : block.begin);
            if (!real) { token_mask[size_t(b * width + t)] = -INFINITY; ragged = true; }
        }
    }
    auto* tokens = constant(runner, options, GGML_TYPE_I32, width, blocks, token_ids);
    auto* masks = ragged ? constant(runner, options, GGML_TYPE_F32, width, blocks, token_mask) : nullptr;
    ggml_tensor* protected_mask = nullptr;
    if (ragged && plan.protected_tokens)
        protected_mask = constant(runner, options, GGML_TYPE_F32, plan.protected_tokens, 1,
                                   std::vector<float>(size_t(plan.protected_tokens), 0));
    // Only 16 * video_blocks entries per query-cell count; no full Q/K padding,
    // no dense sequence mask, no CPU readback of activation-dependent selection.
    ggml_tensor* pair_ids[5] = {};
    ggml_tensor* pair_masks[5] = {};
    for (const auto& block : plan.video) {
        const int qc = int((block.end - block.begin + 15) / 16);
        if (pair_ids[qc]) continue;
        std::vector<int32_t> ids(size_t(16 * blocks), 0);
        std::vector<float> padding(size_t(16 * blocks), -INFINITY);
        for (int64_t b = 0; b < blocks; ++b) {
            int pair = 0;
            for (int a = 0; a < qc; ++a) {
                for (int64_t j = cell_start[size_t(b)]; j < cell_start[size_t(b + 1)]; ++j) {
                    ids[size_t(b * 16 + pair)] = int32_t(a * cell_start.back() + j);
                    padding[size_t(b * 16 + pair++)] = 0;
                }
            }
        }
        pair_ids[qc] = constant(runner, options, GGML_TYPE_I32, 16 * blocks, 1, ids);
        pair_masks[qc] = constant(runner, options, GGML_TYPE_F32, 16, blocks, padding);
    }
    ggml_tensor* output = nullptr;
    std::vector<ggml_tensor*> q_heads, k_heads, v_heads, pooled_k;
    for (int64_t h = 0; h < heads; ++h) {
        auto* qh = ggml_view_2d(c, q, head_dim, q->ne[1], q->nb[1], size_t(h) * q->nb[2]);
        auto* kh = ggml_view_2d(c, k, head_dim, k->ne[1], k->nb[1], size_t(h) * k->nb[2]);
        auto* vh = ggml_view_2d(c, v, head_dim, v->ne[2], v->nb[2], size_t(h) * v->nb[1]);
        q_heads.push_back(qh); k_heads.push_back(kh); v_heads.push_back(vh);
        ggml_tensor* pk = nullptr;
        for (const auto& block : plan.video) pk = append(c, pk, poolmean(c, kh, block), 1);
        pooled_k.push_back(ggml_cont(c, pk));
    }
    for (const auto& chunk : plan.chunks) {
        ggml_tensor* chunk_out = nullptr;
        for (int64_t h = 0; h < heads; ++h) {
            auto* qh = q_heads[size_t(h)];
            auto* kh = k_heads[size_t(h)];
            auto* vh = v_heads[size_t(h)];
            ggml_tensor* selected_k = kh;
            ggml_tensor* selected_v = vh;
            ggml_tensor* validity = nullptr;
            if (chunk.kind == SparseTokenKind::TargetVideo) {
                auto* pq = poolmean(c, qh, {chunk.begin, chunk.end});
                const int qc = int(pq->ne[1]);
                auto* scores = ggml_mul_mat(c, pooled_k[size_t(h)], pq);
                ggml_mul_mat_set_prec(scores, GGML_PREC_F32);
                scores = ggml_scale(c, scores, softmax_scale()); // NOT kv_scale!
                auto* flat = ggml_reshape_2d(c, scores, 1, cell_start.back() * qc);
                auto* pairs = ggml_get_rows(c, flat, pair_ids[qc]);
                pairs = ggml_add(c, ggml_reshape_2d(c, pairs, 16, blocks), pair_masks[qc]);
                auto* maximum = ggml_pool_2d(c, pairs, GGML_OP_POOL_MAX, 16, 1, 16, 1, 0, 0);
                auto* shifted = ggml_sub(c, pairs, ggml_repeat_4d(c, maximum, 16, blocks, 1, 1));
                auto* mass = ggml_sum_rows(c, ggml_exp(c, ggml_cont(c, shifted)));
                auto* lse = ggml_add(c, maximum, ggml_log(c, mass));
                auto* ids = ggml_top_k(c, ggml_reshape_1d(c, lse, blocks), plan.topk);
                ggml_set_name(ids, "h3.subblock.perhead.topk");
                auto* indices = ggml_reshape_1d(c, ggml_get_rows(c, tokens, ids), width * plan.topk);
                selected_k = ggml_get_rows(c, kh, indices);
                selected_v = ggml_get_rows(c, vh, indices);
                if (ragged) validity = ggml_reshape_1d(c, ggml_get_rows(c, masks, ids), width * plan.topk);
                for (const auto& span : layout.spans) {
                    if (span.kind != SparseTokenKind::TargetVideo) {
                        selected_k = append(c, selected_k, ggml_cont(c, slice(c, kh, span.begin, span.end)), 1);
                        selected_v = append(c, selected_v, ggml_cont(c, slice(c, vh, span.begin, span.end)), 1);
                    }
                }
                if (protected_mask) validity = append(c, validity, protected_mask, 0);
            }
            auto* head_out = attention(runner, slice(c, qh, chunk.begin, chunk.end),
                                       selected_k, selected_v, kv_scale, validity);
            if (!head_out) return nullptr;
            chunk_out = append(c, chunk_out, head_out, 0);
        }
        output = append(c, output, chunk_out, 1);
    }
    if (!output || !supported(runner, output, q, k, v)) return nullptr;
    ggml_set_name(output, "h3.subblock.attention");
    return output; // [128 * heads, L, 1, 1], same as ggml_ext_attention_ext
}
#endif // MINIMAX_H3_SPARSE_REFERENCE_ONLY
} // namespace MiniMaxH3
