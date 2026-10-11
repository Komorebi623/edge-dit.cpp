#pragma once
// SPDX-License-Identifier: Apache-2.0
// CPU-only, standard-library mathematical reference. NOT native CUDA, a model
// implementation, or a production feature switch. No shared model/FFN mutation.
// Source: hao-ai-lab/FastVideo @ d5253fd71055927f12aa6c4aa679b93a00238851:
// video_sparse_attn_h3.py (_h3_segment_tile_geometry, _pool_tiles,
// _build_block_mask, MiniMaxH3VSAImpl.forward), video_sparse_attn.py (tile,
// compute_topk), models/dits/minimax_h3.py (to_gate_compress), vsa_guard.py.
// IMPORTANT: this H3 source has a RAW LINEAR gate, NOT a sigmoid gate:
// out = fine_attention + linear(hidden_states) * compressed_attention.
// A sigmoid would incorrectly turn zero-initialized gates into 0.5.
#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <numeric>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace MiniMaxH3 { namespace VSAReference {

inline constexpr bool native_cuda_implemented = false;
inline constexpr bool production_switch_available = false;
inline constexpr size_t invalid_token = std::numeric_limits<size_t>::max();
using Grid = std::array<size_t, 3>;
inline void require(bool ok, const char* message) {
    if (!ok) throw std::invalid_argument(message);
}
inline size_t checked_add(size_t a, size_t b) {
    require(b <= std::numeric_limits<size_t>::max() - a, "VSA size addition overflow");
    return a + b;
}
inline size_t checked_mul(size_t a, size_t b) {
    require(!a || b <= std::numeric_limits<size_t>::max() / a, "VSA size multiplication overflow");
    return a * b;
}
inline size_t volume(const Grid& g) { return checked_mul(checked_mul(g[0], g[1]), g[2]); }
inline Grid tile_shape(size_t elems) {
    if (elems == 64) return {4, 4, 4};
    if (elems == 128) return {4, 4, 8};
    if (elems == 256) return {4, 8, 8};
    throw std::invalid_argument("VSA tile size must be 64, 128, or 256");
}
enum class TokenKind { Text, Audio, DenseReference, Video };
struct Segment {
    TokenKind kind = TokenKind::Text;
    size_t rows = 0;
    Grid grid = {0, 0, 0}; // VIDEO TOKEN grid, not raw latent grid
    static Segment text(size_t n) { return {TokenKind::Text, n, {0, 0, 0}}; }
    static Segment audio(size_t n) { return {TokenKind::Audio, n, {0, 0, 0}}; }
    static Segment reference(size_t n) { return {TokenKind::DenseReference, n, {0, 0, 0}}; }
    static Segment video(Grid g) { return {TokenKind::Video, volume(g), g}; }
};
inline std::vector<Segment> patch_latent_segments(std::vector<Segment> segments, Grid patch) {
    for (size_t p : patch) require(p > 0, "VSA patch axes must be positive");
    for (auto& s : segments) if (s.kind == TokenKind::Video) {
        for (size_t axis = 0; axis < 3; ++axis) {
            require(s.grid[axis] > 0 && s.grid[axis] % patch[axis] == 0,
                    "VSA latent video axes must be positive multiples of patch axes");
            s.grid[axis] /= patch[axis];
        }
        s.rows = volume(s.grid);
    }
    return segments;
}
struct Geometry {
    size_t tile_elems = 256;
    size_t packed_tokens = 0;
    size_t num_prefix_tiles = 0;
    std::vector<Segment> segments;
    std::vector<size_t> variable_block_sizes; // valid tokens compacted to each tile's front
    std::vector<size_t> tile_segment; // pure modality AND pure segment/region
    std::vector<std::pair<size_t, size_t>> video_tile_spans; // [start,end), last = generated
    std::vector<size_t> tile_partition_indices; // compact tile order -> original packed row
    std::vector<size_t> untile_combined_index; // original packed row -> padded tile slot
    std::vector<size_t> source_index; // padded slot -> original row; invalid_token = padding
    size_t tiles() const { return variable_block_sizes.size(); }
    size_t padded_tokens() const { return checked_mul(tiles(), tile_elems); }
};
inline void validate_geometry(const Geometry& g) {
    const Grid shape = tile_shape(g.tile_elems);
    require(g.tiles() > 0 && g.packed_tokens > 0, "VSA empty geometry");
    require(g.tile_segment.size() == g.tiles() && g.num_prefix_tiles <= g.tiles(), "VSA tile metadata shape");
    require(g.source_index.size() == g.padded_tokens() &&
            g.tile_partition_indices.size() == g.packed_tokens &&
            g.untile_combined_index.size() == g.packed_tokens, "VSA index metadata shape");
    std::vector<size_t> starts;
    size_t total = 0, expected_prefix = 0, region = 0, cursor = g.num_prefix_tiles;
    for (size_t s = 0; s < g.segments.size(); ++s) {
        const auto& segment = g.segments[s];
        starts.push_back(total);
        total = checked_add(total, segment.rows);
        if (segment.kind == TokenKind::Video) {
            require(segment.grid[0] && segment.grid[1] && segment.grid[2] &&
                    segment.rows == volume(segment.grid), "VSA video grid/row mismatch");
            Grid counts;
            for (size_t a = 0; a < 3; ++a)
                counts[a] = segment.grid[a] / shape[a] + (segment.grid[a] % shape[a] != 0);
            size_t end = checked_add(cursor, volume(counts));
            require(region < g.video_tile_spans.size() &&
                    g.video_tile_spans[region] == std::make_pair(cursor, end) && end <= g.tiles(),
                    "VSA video spans must cover each region independently after prefix");
            for (size_t t = cursor; t < end; ++t)
                require(g.tile_segment[t] == s, "VSA region tile segment mismatch");
            cursor = end;
            ++region;
        } else {
            require(segment.kind == TokenKind::Text || segment.kind == TokenKind::Audio ||
                    segment.kind == TokenKind::DenseReference, "VSA unknown segment kind");
            size_t n = segment.rows / g.tile_elems + (segment.rows % g.tile_elems != 0);
            size_t end = checked_add(expected_prefix, n);
            require(end <= g.num_prefix_tiles, "VSA dense prefix tile count mismatch");
            for (size_t t = expected_prefix; t < end; ++t)
                require(g.tile_segment[t] == s, "VSA prefix tile crosses a dense segment");
            expected_prefix = end;
        }
    }
    require(total == g.packed_tokens && expected_prefix == g.num_prefix_tiles &&
            region > 0 && region == g.video_tile_spans.size() && cursor == g.tiles(), "VSA geometry coverage mismatch");
    std::vector<uint8_t> seen(total, 0);
    size_t compact = 0;
    for (size_t t = 0; t < g.tiles(); ++t) {
        const size_t count = g.variable_block_sizes[t], s = g.tile_segment[t];
        require(count > 0 && count <= g.tile_elems && s < g.segments.size(), "VSA invalid tile size/segment");
        for (size_t p = 0; p < g.tile_elems; ++p) {
            size_t slot = t * g.tile_elems + p, original = g.source_index[slot];
            if (p >= count) {
                require(original == invalid_token, "VSA padding contains an original token");
                continue;
            }
            require(original < total && !seen[original] && original >= starts[s] &&
                    original - starts[s] < g.segments[s].rows, "VSA noninjective/cross-segment tile map");
            seen[original] = 1;
            require(compact < total && g.tile_partition_indices[compact++] == original &&
                    g.untile_combined_index[original] == slot, "VSA untile is not the exact inverse");
        }
    }
    require(compact == total, "VSA dropped tokens");
}
inline Geometry make_geometry(const std::vector<Segment>& segments, size_t tile_elems = 256) {
    Geometry g;
    g.tile_elems = tile_elems;
    g.segments = segments;
    const Grid shape = tile_shape(tile_elems);
    std::vector<size_t> starts;
    for (const auto& s : segments) {
        starts.push_back(g.packed_tokens);
        if (s.kind == TokenKind::Video)
            require(s.grid[0] && s.grid[1] && s.grid[2] && s.rows == volume(s.grid), "VSA invalid video grid");
        else require(s.kind == TokenKind::Text || s.kind == TokenKind::Audio ||
                     s.kind == TokenKind::DenseReference, "VSA unknown segment kind");
        g.packed_tokens = checked_add(g.packed_tokens, s.rows);
    }
    g.untile_combined_index.assign(g.packed_tokens, invalid_token);
    auto append_tile = [&](size_t segment, const std::vector<size_t>& rows) {
        require(!rows.empty() && rows.size() <= tile_elems, "VSA empty/oversized tile");
        const size_t start = g.source_index.size();
        g.source_index.resize(checked_add(start, tile_elems), invalid_token);
        g.variable_block_sizes.push_back(rows.size());
        g.tile_segment.push_back(segment);
        for (size_t p = 0; p < rows.size(); ++p) {
            g.source_index[start + p] = rows[p];
            g.untile_combined_index[rows[p]] = start + p;
            g.tile_partition_indices.push_back(rows[p]);
        }
    };
    // Dense tiles never straddle two segments, even if adjacent and very short.
    for (size_t s = 0; s < segments.size(); ++s) if (segments[s].kind != TokenKind::Video) {
        for (size_t pos = 0; pos < segments[s].rows;) {
            size_t count = std::min(tile_elems, segments[s].rows - pos);
            std::vector<size_t> rows(count);
            std::iota(rows.begin(), rows.end(), starts[s] + pos);
            append_tile(s, rows);
            pos += count;
        }
    }
    g.num_prefix_tiles = g.tiles();
    for (size_t s = 0; s < segments.size(); ++s) if (segments[s].kind == TokenKind::Video) {
        const Grid grid = segments[s].grid;
        size_t first = g.tiles();
        // Clipped slices flatten in (t,h,w) order, then pack at tile front.
        // Short 3-D tiles do NOT leave spatial holes within their valid prefix.
        for (size_t t = 0; t < grid[0];) {
            size_t dt = std::min(shape[0], grid[0] - t);
            for (size_t h = 0; h < grid[1];) {
                size_t dh = std::min(shape[1], grid[1] - h);
                for (size_t w = 0; w < grid[2];) {
                    size_t dw = std::min(shape[2], grid[2] - w);
                    std::vector<size_t> rows;
                    for (size_t a = t; a < t + dt; ++a)
                        for (size_t b = h; b < h + dh; ++b)
                            for (size_t c = w; c < w + dw; ++c)
                                rows.push_back(starts[s] + (a * grid[1] + b) * grid[2] + c);
                    append_tile(s, rows);
                    w += dw;
                }
                h += dh;
            }
            t += dt;
        }
        g.video_tile_spans.emplace_back(first, g.tiles());
    }
    validate_geometry(g);
    return g;
}

// Contiguous [batch, token, head, channel], consistent for Q/K/V/gate/output.
// Not the ggml model's native tensor layout. No CUDA or ggml dependency.
struct Tensor {
    size_t batch = 0, tokens = 0, heads = 0, dim = 0;
    std::vector<float> data;
    Tensor() = default;
    Tensor(size_t b, size_t t, size_t h, size_t d)
        : batch(b), tokens(t), heads(h), dim(d),
          data(checked_mul(checked_mul(checked_mul(b, t), h), d), 0.0f) {
        require(b && t && h && d, "VSA tensor axes must be positive");
    }
    size_t index(size_t b, size_t t, size_t h, size_t d) const {
        return ((b * tokens + t) * heads + h) * dim + d;
    }
    float& at(size_t b, size_t t, size_t h, size_t d) { return data[index(b, t, h, d)]; }
    float at(size_t b, size_t t, size_t h, size_t d) const { return data[index(b, t, h, d)]; }
};
inline void validate_tensor(const Tensor& x) {
    require(x.batch && x.tokens && x.heads && x.dim &&
            x.data.size() == checked_mul(checked_mul(checked_mul(x.batch, x.tokens), x.heads), x.dim),
            "VSA tensor shape/storage mismatch");
}
inline bool same_shape(const Tensor& a, const Tensor& b) {
    return a.batch == b.batch && a.tokens == b.tokens && a.heads == b.heads && a.dim == b.dim;
}
inline void finite(float x) { require(std::isfinite(x), "VSA nonfinite input or arithmetic overflow"); }
inline void validate_tiled_tensor(const Tensor& x, const Geometry& g) {
    validate_tensor(x);
    bool pair = (g.tile_elems == 64 || g.tile_elems == 128) && g.tiles() % 2 != 0;
    require(x.tokens == g.padded_tokens() ||
            (pair && x.tokens == checked_add(g.padded_tokens(), g.tile_elems)), "VSA invalid logical/dummy length");
    // Invalid slots, including an optional zero-valid transport partner, never
    // participate in pooling, selection, softmax, compression, or untile.
    for (size_t b = 0; b < x.batch; ++b)
        for (size_t t = 0; t < g.padded_tokens(); ++t) if (g.source_index[t] != invalid_token)
            for (size_t h = 0; h < x.heads; ++h)
                for (size_t d = 0; d < x.dim; ++d) finite(x.at(b, t, h, d));
}
inline Tensor tile(const Tensor& packed, const Geometry& g, bool append_dummy_partner = false) {
    validate_geometry(g);
    validate_tensor(packed);
    require(packed.tokens == g.packed_tokens, "VSA packed sequence length mismatch");
    size_t padded = g.padded_tokens();
    if (append_dummy_partner) {
        require((g.tile_elems == 64 || g.tile_elems == 128) && g.tiles() % 2 != 0,
                "VSA dummy partner allowed only for odd logical tile-64/128 geometry");
        padded = checked_add(padded, g.tile_elems);
    }
    Tensor out(packed.batch, padded, packed.heads, packed.dim);
    for (size_t b = 0; b < packed.batch; ++b)
        for (size_t t = 0; t < packed.tokens; ++t)
            for (size_t h = 0; h < packed.heads; ++h)
                for (size_t d = 0; d < packed.dim; ++d) {
                    finite(packed.at(b, t, h, d));
                    out.at(b, g.untile_combined_index[t], h, d) = packed.at(b, t, h, d);
                }
    return out;
}
inline Tensor untile(const Tensor& padded, const Geometry& g) {
    validate_geometry(g);
    validate_tiled_tensor(padded, g);
    Tensor out(padded.batch, g.packed_tokens, padded.heads, padded.dim);
    for (size_t b = 0; b < out.batch; ++b)
        for (size_t t = 0; t < out.tokens; ++t)
            for (size_t h = 0; h < out.heads; ++h)
                for (size_t d = 0; d < out.dim; ++d)
                    out.at(b, t, h, d) = padded.at(b, g.untile_combined_index[t], h, d);
    return out;
}
inline Tensor pool_tiles(const Tensor& x, const Geometry& g) {
    validate_geometry(g);
    validate_tiled_tensor(x, g);
    Tensor out(x.batch, g.tiles(), x.heads, x.dim);
    for (size_t b = 0; b < x.batch; ++b)
        for (size_t t = 0; t < g.tiles(); ++t)
            for (size_t h = 0; h < x.heads; ++h)
                for (size_t d = 0; d < x.dim; ++d) {
                    float sum = 0;
                    for (size_t p = 0; p < g.variable_block_sizes[t]; ++p)
                        sum += x.at(b, t * g.tile_elems + p, h, d);
                    float mean = sum / static_cast<float>(g.variable_block_sizes[t]);
                    finite(mean);
                    out.at(b, t, h, d) = mean;
                }
    return out;
}
struct Scores {
    size_t batch = 0, heads = 0, tiles = 0;
    std::vector<float> data; // [B,H,query_tile,key_tile]
    size_t index(size_t b, size_t h, size_t q, size_t k) const {
        return ((b * heads + h) * tiles + q) * tiles + k;
    }
    float at(size_t b, size_t h, size_t q, size_t k) const { return data[index(b, h, q, k)]; }
};
inline Scores pooled_scores(const Tensor& q, const Tensor& k, const Geometry& g) {
    require(same_shape(q, k), "VSA Q/K shapes differ");
    Tensor qp = pool_tiles(q, g), kp = pool_tiles(k, g);
    Scores scores{q.batch, q.heads, g.tiles(), {}};
    scores.data.resize(checked_mul(checked_mul(checked_mul(q.batch, q.heads), g.tiles()), g.tiles()));
    const float scale = 1.0f / std::sqrt(static_cast<float>(q.dim));
    for (size_t b = 0; b < q.batch; ++b)
        for (size_t h = 0; h < q.heads; ++h)
            for (size_t i = 0; i < g.tiles(); ++i)
                for (size_t j = 0; j < g.tiles(); ++j) {
                    float dot = 0;
                    for (size_t d = 0; d < q.dim; ++d) dot += qp.at(b, i, h, d) * kp.at(b, j, h, d);
                    float score = dot * scale;
                    finite(score);
                    scores.data[scores.index(b, h, i, j)] = score;
                }
    return scores;
}
inline size_t compute_topk(double sparsity, size_t blocks) {
    require(blocks > 0 && std::isfinite(sparsity), "VSA invalid top-k sparsity/block count");
    if (sparsity <= 0) return blocks;
    if (sparsity >= 1) return 1;
    // Equivalent to FastVideo max(1,min(ceil((1-sparsity)*blocks),blocks)).
    double raw = std::ceil((1.0 - sparsity) * static_cast<double>(blocks));
    if (raw >= static_cast<double>(blocks)) return blocks;
    return std::max(size_t(1), static_cast<size_t>(raw));
}
struct RegionBudget {
    bool exact = false;
    size_t count = 0;
    double sparsity = 0;
    static RegionBudget topk(size_t n) { return {true, n, 0}; }
    static RegionBudget sparse(double s) { return {false, 0, s}; }
};
struct RoutingOptions {
    double sparsity = 0.9; // generated region; <=0 forces ALL regions dense
    std::optional<double> reference_keep_rate;
    std::vector<RegionBudget> region_budgets; // optional explicit per-region CPU experiment
    bool exempt = true; // default: forced dense prefix keys, dense non-video queries
    bool force_dense = false; // dense schedule opt-out; does NOT disable an active gate
};
struct Routing {
    Scores scores;
    std::vector<uint8_t> mask; // [B,H,query_tile,key_tile]; no token deletion
    std::vector<size_t> region_topk;
    bool keeps(size_t b, size_t h, size_t q, size_t k) const {
        return mask[scores.index(b, h, q, k)] != 0;
    }
};
inline Routing route(const Scores& scores, const Geometry& g, const RoutingOptions& options) {
    validate_geometry(g);
    require(scores.batch && scores.heads && scores.tiles == g.tiles() &&
            scores.data.size() == checked_mul(checked_mul(checked_mul(scores.batch, scores.heads),
                                                         scores.tiles), scores.tiles), "VSA score shape mismatch");
    for (float s : scores.data) finite(s);
    require(std::isfinite(options.sparsity), "VSA nonfinite target sparsity");
    if (options.reference_keep_rate)
        require(std::isfinite(*options.reference_keep_rate) && *options.reference_keep_rate >= 0 &&
                *options.reference_keep_rate <= 1, "VSA reference keep rate outside [0,1]");
    size_t regions = g.video_tile_spans.size();
    require(options.exempt || regions == 1, "VSA compete mode supports only one video region");
    require(options.region_budgets.empty() || options.region_budgets.size() == regions,
            "VSA requires one explicit budget per video region");
    require(regions == 1 || !options.region_budgets.empty() || options.reference_keep_rate.has_value(),
            "VSA multiple regions require a reference keep rate or explicit per-region budgets");
    Routing out{scores, std::vector<uint8_t>(scores.data.size(), 0), {}};
    bool all_keep = true;
    for (size_t r = 0; r < regions; ++r) {
        size_t n = g.video_tile_spans[r].second - g.video_tile_spans[r].first;
        size_t count;
        if (!options.region_budgets.empty()) {
            const auto& budget = options.region_budgets[r];
            if (budget.exact) {
                require(budget.count > 0 && budget.count <= n, "VSA explicit top-k outside [1,region_tiles]");
                count = budget.count;
            } else count = compute_topk(budget.sparsity, n);
        } else count = compute_topk(r + 1 == regions ? options.sparsity : 1.0 - *options.reference_keep_rate, n);
        if (options.force_dense || options.sparsity <= 0) count = n;
        out.region_topk.push_back(count);
        all_keep = all_keep && count == n;
    }
    if (all_keep) {
        std::fill(out.mask.begin(), out.mask.end(), uint8_t(1));
        return out;
    }
    for (size_t b = 0; b < scores.batch; ++b)
        for (size_t h = 0; h < scores.heads; ++h)
            for (size_t q = 0; q < g.tiles(); ++q) {
                auto set = [&](size_t k) { out.mask[scores.index(b, h, q, k)] = 1; };
                if (q < g.num_prefix_tiles) {
                    for (size_t k = 0; k < g.tiles(); ++k) set(k);
                    continue;
                }
                auto select = [&](size_t first, size_t end, size_t count) {
                    std::vector<size_t> indices(end - first);
                    std::iota(indices.begin(), indices.end(), first);
                    // CPU ties use lower key tile ID; torch.topk tie ordering
                    // is not specified. Do not claim bitwise GPU mask parity on ties.
                    std::partial_sort(indices.begin(), indices.begin() + count, indices.end(),
                        [&](size_t a, size_t c) {
                            float av = scores.at(b, h, q, a), cv = scores.at(b, h, q, c);
                            return av == cv ? a < c : av > cv;
                        });
                    for (size_t rank = 0; rank < count; ++rank) set(indices[rank]);
                };
                if (options.exempt) {
                    for (size_t k = 0; k < g.num_prefix_tiles; ++k) set(k);
                    for (size_t r = 0; r < regions; ++r)
                        select(g.video_tile_spans[r].first, g.video_tile_spans[r].second, out.region_topk[r]);
                } else select(0, g.tiles(), std::min(checked_add(out.region_topk[0], g.num_prefix_tiles), g.tiles()));
            }
    return out;
}
inline std::vector<float> softmax(const std::vector<float>& logits) {
    require(!logits.empty(), "VSA empty attention key set");
    for (float l : logits) finite(l);
    const float maximum = *std::max_element(logits.begin(), logits.end());
    std::vector<float> weights(logits.size());
    float sum = 0;
    for (size_t i = 0; i < logits.size(); ++i) {
        weights[i] = std::exp(logits[i] - maximum);
        sum += weights[i];
    }
    finite(sum);
    require(sum > 0, "VSA invalid softmax normalizer");
    for (float& w : weights) w /= sum;
    return weights;
}
inline Tensor fine_attention(const Tensor& q, const Tensor& k, const Tensor& v,
                             const Geometry& g, const Routing& routing) {
    validate_geometry(g);
    validate_tiled_tensor(q, g);
    validate_tiled_tensor(k, g);
    validate_tiled_tensor(v, g);
    require(same_shape(q, k) && same_shape(q, v), "VSA Q/K/V shapes differ");
    require(routing.scores.batch == q.batch && routing.scores.heads == q.heads &&
            routing.scores.tiles == g.tiles() && routing.mask.size() ==
            checked_mul(checked_mul(checked_mul(q.batch, q.heads), g.tiles()), g.tiles()), "VSA routing shape mismatch");
    Tensor out(q.batch, q.tokens, q.heads, q.dim);
    const float scale = 1.0f / std::sqrt(static_cast<float>(q.dim));
    for (size_t b = 0; b < q.batch; ++b)
        for (size_t h = 0; h < q.heads; ++h)
            for (size_t qt = 0; qt < g.tiles(); ++qt) {
                std::vector<size_t> keys;
                for (size_t kt = 0; kt < g.tiles(); ++kt) if (routing.keeps(b, h, qt, kt))
                    for (size_t p = 0; p < g.variable_block_sizes[kt]; ++p) keys.push_back(kt * g.tile_elems + p);
                require(!keys.empty(), "VSA route selected no valid keys");
                for (size_t p = 0; p < g.variable_block_sizes[qt]; ++p) {
                    size_t query = qt * g.tile_elems + p;
                    std::vector<float> logits(keys.size());
                    for (size_t j = 0; j < keys.size(); ++j) {
                        float dot = 0;
                        for (size_t d = 0; d < q.dim; ++d) dot += q.at(b, query, h, d) * k.at(b, keys[j], h, d);
                        logits[j] = dot * scale;
                    }
                    auto weights = softmax(logits);
                    for (size_t d = 0; d < q.dim; ++d) {
                        float sum = 0;
                        for (size_t j = 0; j < keys.size(); ++j) sum += weights[j] * v.at(b, keys[j], h, d);
                        finite(sum);
                        out.at(b, query, h, d) = sum;
                    }
                }
            }
    return out;
}
inline Tensor compressed_attention(const Tensor& v, const Geometry& g, const Scores& scores) {
    Tensor vp = pool_tiles(v, g);
    require(scores.batch == v.batch && scores.heads == v.heads && scores.tiles == g.tiles() &&
            scores.data.size() == checked_mul(checked_mul(checked_mul(v.batch, v.heads), g.tiles()), g.tiles()),
            "VSA compression score shape mismatch");
    Tensor out(v.batch, g.tiles(), v.heads, v.dim);
    for (size_t b = 0; b < v.batch; ++b)
        for (size_t h = 0; h < v.heads; ++h)
            for (size_t q = 0; q < g.tiles(); ++q) {
                std::vector<float> logits(g.tiles());
                for (size_t k = 0; k < g.tiles(); ++k) logits[k] = scores.at(b, h, q, k);
                auto weights = softmax(logits); // ALL logical tiles, not selected-only or discarded-only
                for (size_t d = 0; d < v.dim; ++d) {
                    float sum = 0;
                    for (size_t k = 0; k < g.tiles(); ++k) sum += weights[k] * vp.at(b, k, h, d);
                    finite(sum);
                    out.at(b, q, h, d) = sum;
                }
            }
    return out;
}
inline Tensor fuse_compression(const Tensor& fine, const Tensor& compressed, const Tensor& raw_gate,
                               const Geometry& g) {
    validate_geometry(g);
    validate_tiled_tensor(fine, g);
    validate_tiled_tensor(raw_gate, g);
    validate_tensor(compressed);
    require(same_shape(fine, raw_gate) && compressed.batch == fine.batch && compressed.heads == fine.heads &&
            compressed.dim == fine.dim && compressed.tokens == g.tiles(), "VSA gate/compression shape mismatch");
    Tensor out(fine.batch, fine.tokens, fine.heads, fine.dim);
    for (size_t b = 0; b < fine.batch; ++b)
        for (size_t t = 0; t < g.padded_tokens(); ++t) if (g.source_index[t] != invalid_token)
            for (size_t h = 0; h < fine.heads; ++h)
                for (size_t d = 0; d < fine.dim; ++d) {
                    float c = compressed.at(b, t / g.tile_elems, h, d);
                    finite(c);
                    float y = fine.at(b, t, h, d) + raw_gate.at(b, t, h, d) * c;
                    finite(y);
                    out.at(b, t, h, d) = y;
                }
    return out;
}
// W is [heads*head_dim, hidden_dim], row-major, bias=False as in FastVideo.
// hidden is [B,packed_tokens,1,hidden_dim]. Gate is per token/head/channel.
inline Tensor linear_gate(const Tensor& hidden, const std::vector<float>& weight, size_t heads, size_t head_dim) {
    validate_tensor(hidden);
    require(hidden.heads == 1 && heads && head_dim &&
            weight.size() == checked_mul(checked_mul(heads, head_dim), hidden.dim), "VSA linear gate weight shape");
    for (float x : hidden.data) finite(x);
    for (float x : weight) finite(x);
    Tensor out(hidden.batch, hidden.tokens, heads, head_dim);
    for (size_t b = 0; b < hidden.batch; ++b)
        for (size_t t = 0; t < hidden.tokens; ++t)
            for (size_t h = 0; h < heads; ++h)
                for (size_t d = 0; d < head_dim; ++d) {
                    float sum = 0;
                    for (size_t c = 0; c < hidden.dim; ++c)
                        sum += weight[(h * head_dim + d) * hidden.dim + c] * hidden.at(b, t, 0, c);
                    finite(sum);
                    out.at(b, t, h, d) = sum; // deliberately no sigmoid/clipping/bias
                }
    return out;
}
struct AttentionResult {
    Tensor output; // original packed order, every original token remains
    Tensor fine; // original packed order before gate merge
    Tensor compressed_tiles; // [B,n_logical_tiles,H,D]; empty when gate omitted
    Routing routing;
};
inline AttentionResult attention(const Tensor& q, const Tensor& k, const Tensor& v, const Geometry& g,
                                 const RoutingOptions& options, const Tensor* packed_raw_gate = nullptr,
                                 bool append_dummy_partner = false) {
    require(same_shape(q, k) && same_shape(q, v), "VSA packed Q/K/V shapes differ");
    if (packed_raw_gate) require(same_shape(q, *packed_raw_gate), "VSA packed gate shape mismatch");
    Tensor tq = tile(q, g, append_dummy_partner), tk = tile(k, g, append_dummy_partner),
           tv = tile(v, g, append_dummy_partner);
    AttentionResult out;
    out.routing = route(pooled_scores(tq, tk, g), g, options);
    Tensor fine = fine_attention(tq, tk, tv, g, out.routing);
    out.fine = untile(fine, g);
    if (packed_raw_gate) {
        Tensor gate = tile(*packed_raw_gate, g, append_dummy_partner);
        out.compressed_tiles = compressed_attention(tv, g, out.routing.scores);
        out.output = untile(fuse_compression(fine, out.compressed_tiles, gate, g), g);
    } else out.output = out.fine;
    return out;
}

// Independent, read-only checkpoint capability inspection. It does not load,
// patch, zero-initialize, activate, or mutate model weights. Caller must provide
// every main-attention gate with the expected checkpoint tensor name/shape.
// Nonzero is only a structural check, NOT proof of training/quality/provenance.
struct GateWeightView {
    std::string tensor_name;
    size_t input_dim = 0, output_dim = 0;
    const std::vector<float>* f32 = nullptr;
    const std::vector<uint8_t>* packed_nvfp4 = nullptr;
};
enum class WeightCapability { Missing, Invalid, ZeroOnly, FloatReference, PackedInspectionOnly };
struct CapabilityReport {
    WeightCapability capability = WeightCapability::Missing;
    bool complete = false;
    bool any_nonzero_gate = false;
    bool cpu_linear_gate_supported = false;
    bool production_ready = false; // always false: no native backend/real-weight qualification here
    std::string reason;
};
inline CapabilityReport inspect_gate_weights(const std::vector<GateWeightView>& gates,
                                            size_t layers, size_t hidden_dim, size_t heads, size_t head_dim) {
    CapabilityReport out;
    if (!layers || !hidden_dim || !heads || !head_dim || gates.size() != layers) {
        out.reason = "Missing full FastH3 to_gate_compress layer coverage; base H3 is not VSA-qualified";
        return out;
    }
    size_t output_dim, elements;
    try { output_dim = checked_mul(heads, head_dim); elements = checked_mul(output_dim, hidden_dim); }
    catch (const std::invalid_argument&) {
        out.capability = WeightCapability::Invalid;
        out.reason = "Gate architecture size overflow";
        return out;
    }
    bool packed_seen = false;
    for (size_t layer = 0; layer < layers; ++layer) {
        const auto& gate = gates[layer];
        std::string name = "transformer_blocks." + std::to_string(layer) + ".attn.to_gate_compress.weight";
        if (gate.tensor_name != name || gate.input_dim != hidden_dim || gate.output_dim != output_dim ||
            (gate.f32 && gate.packed_nvfp4)) {
            out.capability = WeightCapability::Invalid;
            out.reason = "Gate checkpoint tensor name, shape, or encoding mismatches FastH3 architecture";
            return out;
        }
        if (gate.f32 && !gate.f32->empty()) {
            if (gate.f32->size() != elements ||
                !std::all_of(gate.f32->begin(), gate.f32->end(), [](float x) { return std::isfinite(x); })) {
                out.capability = WeightCapability::Invalid;
                out.reason = "Gate float weight storage is incomplete or nonfinite";
                return out;
            }
            out.any_nonzero_gate = out.any_nonzero_gate ||
                std::any_of(gate.f32->begin(), gate.f32->end(), [](float x) { return x != 0.0f; });
        } else if (gate.packed_nvfp4 && !gate.packed_nvfp4->empty()) {
            if (hidden_dim % 2 != 0 || gate.packed_nvfp4->size() != elements / 2) {
                out.capability = WeightCapability::Invalid;
                out.reason = "NVFP4 gate packed shape mismatch";
                return out;
            }
            packed_seen = true;
            out.any_nonzero_gate = out.any_nonzero_gate ||
                std::any_of(gate.packed_nvfp4->begin(), gate.packed_nvfp4->end(),
                            [](uint8_t x) { return (x & 0x77) != 0; }); // both signed zeros are zero
        } else {
            out.reason = "Missing/unmaterialized gate weights; defer until actual checkpoint weights are available";
            return out;
        }
    }
    out.complete = true;
    if (!out.any_nonzero_gate) {
        out.capability = WeightCapability::ZeroOnly;
        out.reason = "Every to_gate_compress weight is zero; refuse VSA production, allow explicit pure-sparse CPU oracle";
    } else if (packed_seen) {
        out.capability = WeightCapability::PackedInspectionOnly;
        out.reason = "Nonzero NVFP4 gate codes inspected only; scale/dequantization and real-weight parity are not implemented";
    } else {
        out.capability = WeightCapability::FloatReference;
        out.cpu_linear_gate_supported = true;
        out.reason = "Complete nonzero FastH3 float gate shapes permit CPU reference only; no production qualification";
    }
    return out;
}

}} // namespace MiniMaxH3::VSAReference
