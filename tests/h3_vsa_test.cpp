// CPU-only tests. No GPU probing, torch, ggml, model loading or FFN changes.
#include "dit_models/models/minimax_h3_vsa_reference.hpp"
#include <functional>
#include <iomanip>
#include <iostream>
#include <random>
#include <sstream>

using namespace MiniMaxH3::VSAReference;
namespace {
size_t checks = 0;
void check(bool condition, const char* message) {
    ++checks;
    if (!condition) throw std::runtime_error(message);
}
void close(float actual, float expected, const char* message, float tolerance = 2e-5f) {
    check(std::isfinite(actual) && std::isfinite(expected) &&
          std::abs(actual - expected) <= tolerance * std::max(1.0f, std::abs(expected)), message);
}
void tensor_close(const Tensor& actual, const Tensor& expected, const char* message, float tolerance = 2e-5f) {
    check(same_shape(actual, expected), message);
    for (size_t i = 0; i < actual.data.size(); ++i) close(actual.data[i], expected.data[i], message, tolerance);
}
void rejects(const std::function<void()>& f, const char* message) {
    bool rejected = false;
    try { f(); } catch (const std::invalid_argument&) { rejected = true; }
    check(rejected, message);
}
Tensor sample(size_t batch, size_t tokens, size_t heads, size_t dim, uint32_t seed) {
    Tensor x(batch, tokens, heads, dim);
    std::mt19937 random(seed);
    for (float& f : x.data) f = (static_cast<float>(random() % 2001) - 1000) / 1500.0f;
    return x;
}
// Independent packed-row softmax oracle. Deliberately does NOT call tile,
// pool_tiles, fine_attention, softmax or compression. Long-double reduction
// separates the numerical baseline from the reference's F32 accumulation.
Tensor packed_oracle(const Tensor& q, const Tensor& k, const Tensor& v,
                     const Geometry& g, const Routing* routing = nullptr) {
    Tensor out(q.batch, q.tokens, q.heads, q.dim);
    for (size_t b = 0; b < q.batch; ++b)
        for (size_t h = 0; h < q.heads; ++h)
            for (size_t row = 0; row < q.tokens; ++row) {
                std::vector<size_t> keys;
                std::vector<long double> logits;
                for (size_t key = 0; key < k.tokens; ++key) {
                    if (routing && !routing->keeps(b, h, g.untile_combined_index[row] / g.tile_elems,
                                                        g.untile_combined_index[key] / g.tile_elems)) continue;
                    long double score = 0;
                    for (size_t d = 0; d < q.dim; ++d)
                        score += static_cast<long double>(q.at(b, row, h, d)) * k.at(b, key, h, d);
                    logits.push_back(score / std::sqrt(static_cast<long double>(q.dim)));
                    keys.push_back(key);
                }
                long double maximum = *std::max_element(logits.begin(), logits.end()), total = 0;
                for (long double& l : logits) { l = std::exp(l - maximum); total += l; }
                for (size_t d = 0; d < q.dim; ++d) {
                    long double sum = 0;
                    for (size_t j = 0; j < keys.size(); ++j)
                        sum += logits[j] / total * v.at(b, keys[j], h, d);
                    out.at(b, row, h, d) = static_cast<float>(sum);
                }
            }
    return out;
}
// Independent clipped-3D ordering (sort original tokens by tile coordinate,
// then by original row), rather than repeating the implementation's loops.
std::vector<size_t> independent_partition(const std::vector<Segment>& segments, size_t elems) {
    Grid shape = tile_shape(elems);
    std::vector<size_t> dense, video;
    size_t offset = 0;
    for (const auto& segment : segments) {
        if (segment.kind != TokenKind::Video) {
            for (size_t row = 0; row < segment.rows; ++row) dense.push_back(offset + row);
        } else {
            std::vector<size_t> local(segment.rows);
            std::iota(local.begin(), local.end(), size_t(0));
            auto key = [&](size_t row) {
                size_t t = row / (segment.grid[1] * segment.grid[2]);
                size_t h = (row / segment.grid[2]) % segment.grid[1], w = row % segment.grid[2];
                return Grid{t / shape[0], h / shape[1], w / shape[2]};
            };
            std::stable_sort(local.begin(), local.end(), [&](size_t a, size_t b) { return key(a) < key(b); });
            for (size_t row : local) video.push_back(offset + row);
        }
        offset += segment.rows;
    }
    dense.insert(dense.end(), video.begin(), video.end());
    return dense;
}
void geometry_tests() {
    check(tile_shape(64) == Grid{4, 4, 4}, "video64 shape");
    check(tile_shape(128) == Grid{4, 4, 8}, "video128 shape");
    check(tile_shape(256) == Grid{4, 8, 8}, "video256 shape");
    for (size_t elems : {64u, 128u, 256u}) {
        std::vector<Segment> segments{Segment::text(elems + 3), Segment::video({5, 5, 9}),
                                     Segment::audio(2), Segment::reference(1), Segment::text(0),
                                     Segment::video({1, 3, 2}), Segment::audio(elems - 1)};
        Geometry g = make_geometry(segments, elems);
        check(g.num_prefix_tiles == 5, "dense segment tails were joined");
        check(g.video_tile_spans.size() == 2 && g.video_tile_spans[0].first == 5 &&
              g.video_tile_spans[0].second == g.video_tile_spans[1].first &&
              g.video_tile_spans[1].second == g.tiles(), "interleaved regions not contiguous after prefix");
        check(g.variable_block_sizes[0] == elems && g.variable_block_sizes[1] == 3 &&
              g.variable_block_sizes[2] == 2 && g.variable_block_sizes[3] == 1 &&
              g.variable_block_sizes[4] == elems - 1 && g.variable_block_sizes.back() == 6,
              "short dense/video tails padded as valid tokens");
        check(g.tile_partition_indices == independent_partition(segments, elems), "3D flatten/tile order mismatch");
        check(std::accumulate(g.variable_block_sizes.begin(), g.variable_block_sizes.end(), size_t(0)) ==
              g.packed_tokens, "VSA dropped tokens");
        Tensor input = sample(2, g.packed_tokens, 2, 3, uint32_t(elems));
        const auto before = input.data;
        Tensor padded = tile(input, g);
        tensor_close(untile(padded, g), input, "tile/untile is not exact", 0);
        check(input.data == before, "tile mutated original sequence");
        for (size_t b = 0; b < padded.batch; ++b)
            for (size_t p = 0; p < padded.tokens; ++p) if (g.source_index[p] == invalid_token)
                for (size_t h = 0; h < padded.heads; ++h)
                    for (size_t d = 0; d < padded.dim; ++d) check(padded.at(b, p, h, d) == 0, "tile padding not zero");
        Tensor means = pool_tiles(padded, g);
        for (size_t b = 0; b < means.batch; ++b)
            for (size_t t = 0; t < g.tiles(); ++t)
                for (size_t h = 0; h < means.heads; ++h)
                    for (size_t d = 0; d < means.dim; ++d) {
                        long double sum = 0;
                        for (size_t p = 0; p < g.variable_block_sizes[t]; ++p)
                            sum += input.at(b, g.source_index[t * elems + p], h, d);
                        close(means.at(b, t, h, d), static_cast<float>(sum / g.variable_block_sizes[t]),
                              "pool divided by padded size instead of true valid count");
                    }
    }
    Geometry tiny = make_geometry({Segment::text(1), Segment::audio(1), Segment::reference(1),
                                  Segment::video({1, 1, 1})}, 64);
    check(tiny.variable_block_sizes == std::vector<size_t>({1, 1, 1, 1}), "single-token modality purity");
    auto patched = patch_latent_segments({Segment::text(3), Segment::video({5, 10, 18}), Segment::audio(2)}, {1, 2, 2});
    check(patched[1].grid == Grid{5, 5, 9} && patched[1].rows == 225 && patched[0].rows == 3,
          "latent-to-token patch conversion");
    rejects([] { patch_latent_segments({Segment::video({5, 9, 18})}, {1, 2, 2}); }, "nonmultiple patch accepted");
    rejects([] { patch_latent_segments({Segment::video({5, 10, 18})}, {1, 0, 2}); }, "zero patch accepted");
    rejects([] { make_geometry({Segment::video({0, 1, 1})}); }, "zero video axis accepted");
    rejects([] { make_geometry({Segment::text(1)}); }, "no-video geometry accepted");
    rejects([] { make_geometry({}); }, "empty geometry accepted");
    rejects([] { tile_shape(32); }, "unsupported tile size accepted");
    rejects([] { checked_mul(std::numeric_limits<size_t>::max(), 2); }, "multiplication overflow accepted");
    rejects([] { checked_add(std::numeric_limits<size_t>::max(), 1); }, "addition overflow accepted");
    Geometry bad = tiny;
    bad.untile_combined_index[0] = bad.untile_combined_index[1];
    rejects([&] { validate_geometry(bad); }, "noninverse untile accepted");
    bad = tiny; bad.source_index[1] = 0;
    rejects([&] { validate_geometry(bad); }, "original token in pad accepted");
    bad = tiny; bad.variable_block_sizes[0] = 0;
    rejects([&] { validate_geometry(bad); }, "zero valid logical tile accepted");
    bad = tiny; bad.video_tile_spans[0].first = 2;
    rejects([&] { validate_geometry(bad); }, "overlapping prefix/video span accepted");
    bad = tiny; bad.tile_segment[0] = 1;
    rejects([&] { validate_geometry(bad); }, "cross-segment tile accepted");
}
void routing_tests() {
    check(compute_topk(0.9, 11) == 2 && compute_topk(0.5, 3) == 2 &&
          compute_topk(1, 20) == 1 && compute_topk(-1, 20) == 20 && compute_topk(2, 20) == 1,
          "FastVideo ceil/clamp top-k semantics");
    rejects([] { compute_topk(0.5, 0); }, "zero routing blocks accepted");
    rejects([] { compute_topk(std::numeric_limits<double>::quiet_NaN(), 3); }, "NaN sparsity accepted");
    Geometry g = make_geometry({Segment::video({1, 1, 12}), Segment::text(1), Segment::audio(1),
                               Segment::reference(1), Segment::video({1, 1, 16}), Segment::text(1)}, 64);
    Scores scores{2, 2, g.tiles(), std::vector<float>(2 * 2 * g.tiles() * g.tiles())};
    for (size_t b = 0; b < 2; ++b)
        for (size_t h = 0; h < 2; ++h)
            for (size_t q = 0; q < g.tiles(); ++q)
                for (size_t k = 0; k < g.tiles(); ++k)
                    scores.data[scores.index(b, h, q, k)] =
                        static_cast<float>((k + b + h + q) % g.tiles()); // unique per row
    RoutingOptions options;
    options.region_budgets = {RegionBudget::topk(2), RegionBudget::topk(1)};
    Routing routed = route(scores, g, options);
    check(routed.region_topk == std::vector<size_t>({2, 1}), "explicit per-region top-k ignored");
    for (size_t b = 0; b < 2; ++b)
        for (size_t h = 0; h < 2; ++h)
            for (size_t q = 0; q < g.tiles(); ++q) {
                for (size_t k = 0; k < g.num_prefix_tiles; ++k)
                    check(routed.keeps(b, h, q, k), "text/audio/reference key not forced dense");
                if (q < g.num_prefix_tiles) {
                    for (size_t k = 0; k < g.tiles(); ++k)
                        check(routed.keeps(b, h, q, k), "non-video query was sparsified");
                } else {
                    for (size_t r = 0; r < g.video_tile_spans.size(); ++r) {
                        auto span = g.video_tile_spans[r];
                        std::vector<size_t> ids(span.second - span.first);
                        std::iota(ids.begin(), ids.end(), span.first);
                        std::sort(ids.begin(), ids.end(), [&](size_t a, size_t c) {
                            return scores.at(b, h, q, a) > scores.at(b, h, q, c);
                        });
                        for (size_t rank = 0; rank < ids.size(); ++rank)
                            check(routed.keeps(b, h, q, ids[rank]) == (rank < routed.region_topk[r]),
                                  "top-k was not independent for each batch/head/query/region");
                    }
                }
            }
    check(routed.keeps(0, 0, 4, 6) != routed.keeps(0, 1, 4, 6), "per-head route not head-local");
    check(routed.keeps(0, 0, 4, 6) != routed.keeps(0, 0, 5, 6), "per-query route not query-local");
    check(routed.keeps(0, 0, 4, 6) != routed.keeps(1, 0, 4, 6), "per-batch route not batch-local");
    RoutingOptions defaults; defaults.sparsity = 0.75; defaults.reference_keep_rate = 0.5;
    check(route(scores, g, defaults).region_topk == std::vector<size_t>({2, 1}), "reference keep vs target sparsity");
    defaults.force_dense = true;
    auto dense = route(scores, g, defaults);
    check(std::all_of(dense.mask.begin(), dense.mask.end(), [](uint8_t v) { return v == 1; }), "dense layer failed");
    defaults.force_dense = false; defaults.sparsity = 0; defaults.reference_keep_rate = 0;
    dense = route(scores, g, defaults);
    check(dense.region_topk == std::vector<size_t>({3, 4}), "dense step failed to override reference keep rate");
    options.region_budgets = {RegionBudget::topk(3), RegionBudget::topk(4)};
    dense = route(scores, g, options);
    check(std::all_of(dense.mask.begin(), dense.mask.end(), [](uint8_t v) { return v == 1; }), "all-keep mask not dense");
    options.region_budgets = {RegionBudget::topk(1), RegionBudget::topk(1)};
    std::fill(scores.data.begin(), scores.data.end(), 0);
    auto tied = route(scores, g, options);
    for (const auto& span : g.video_tile_spans)
        check(tied.keeps(0, 0, 4, span.first) && !tied.keeps(0, 0, 4, span.first + 1), "CPU tie policy drift");
    rejects([&] { RoutingOptions x; route(scores, g, x); }, "missing multi-region ref_keep accepted");
    rejects([&] { auto x = options; x.exempt = false; route(scores, g, x); }, "compete accepted multiple regions");
    rejects([&] { auto x = options; x.region_budgets.pop_back(); route(scores, g, x); }, "missing region budget accepted");
    rejects([&] { auto x = options; x.region_budgets[0] = RegionBudget::topk(0); route(scores, g, x); }, "topk zero accepted");
    rejects([&] { auto x = options; x.region_budgets[0] = RegionBudget::topk(4); route(scores, g, x); }, "topk too large accepted");
    rejects([&] { auto x = defaults; x.reference_keep_rate = 1.1; route(scores, g, x); }, "invalid ref keep accepted");
    Scores nonfinite = scores; nonfinite.data[0] = std::numeric_limits<float>::infinity();
    rejects([&] { route(nonfinite, g, options); }, "nonfinite routing scores accepted");
    Geometry single = make_geometry({Segment::text(1), Segment::video({1, 1, 16})}, 64);
    Scores compete_scores{1, 1, single.tiles(), std::vector<float>(single.tiles() * single.tiles(), 0)};
    for (size_t q = 0; q < single.tiles(); ++q)
        for (size_t k = 0; k < single.tiles(); ++k)
            compete_scores.data[compete_scores.index(0, 0, q, k)] = static_cast<float>(k);
    RoutingOptions compete; compete.exempt = false; compete.sparsity = 0.75;
    auto competed = route(compete_scores, single, compete);
    check(!competed.keeps(0, 0, 1, 0) && competed.keeps(0, 0, 1, 3) && competed.keeps(0, 0, 1, 4),
          "single-region compete budget must equal video topk + prefix tiles");
    for (size_t k = 0; k < single.tiles(); ++k) check(competed.keeps(0, 0, 0, k), "compete sparsified prefix query");
}
void attention_tests() {
    for (size_t elems : {64u, 128u, 256u}) {
        Geometry g = make_geometry({Segment::text(2), Segment::video({3, 3, 9}), Segment::audio(3),
                                   Segment::reference(2), Segment::video({1, 2, 9})}, elems);
        Tensor q = sample(2, g.packed_tokens, 2, 3, 101), k = sample(2, g.packed_tokens, 2, 3, 102),
               v = sample(2, g.packed_tokens, 2, 3, 103), gate = sample(2, g.packed_tokens, 2, 3, 104);
        const auto q_before = q.data, k_before = k.data, v_before = v.data, gate_before = gate.data;
        RoutingOptions dense; dense.sparsity = 0; dense.reference_keep_rate = 0;
        Tensor oracle = packed_oracle(q, k, v, g);
        AttentionResult all = attention(q, k, v, g, dense);
        tensor_close(all.output, oracle, "all-keep differs from ordinary dense packed attention");
        check(all.output.tokens == g.packed_tokens && all.compressed_tiles.data.empty(), "all-keep removed tokens/added gate");
        RoutingOptions sparse; sparse.sparsity = 0.9; sparse.reference_keep_rate = 0.1;
        AttentionResult result = attention(q, k, v, g, sparse);
        tensor_close(result.output, packed_oracle(q, k, v, g, &result.routing), "selected-connection fine attention mismatch");
        for (size_t t = 0; t < g.packed_tokens; ++t) if (g.untile_combined_index[t] / elems < g.num_prefix_tiles)
            for (size_t b = 0; b < q.batch; ++b)
                for (size_t h = 0; h < q.heads; ++h)
                    for (size_t d = 0; d < q.dim; ++d)
                        close(result.output.at(b, t, h, d), oracle.at(b, t, h, d),
                              "text/audio/dense-reference output not dense baseline");
        Tensor zero(q.batch, q.tokens, q.heads, q.dim);
        tensor_close(attention(q, k, v, g, sparse, &zero).output, result.output, "zero raw gate != pure sparse", 0);
        AttentionResult gated = attention(q, k, v, g, sparse, &gate);
        tensor_close(gated.fine, result.output, "gate mutated fine-attention baseline", 0);
        check(gated.output.tokens == q.tokens && gated.compressed_tiles.tokens == g.tiles(), "gate branch deletes sequence rows");
        Tensor qp = pool_tiles(tile(q, g), g), kp = pool_tiles(tile(k, g), g), vp = pool_tiles(tile(v, g), g);
        // Analytic long-double pooled softmax weighted V, over every tile.
        for (size_t b = 0; b < q.batch; ++b)
            for (size_t h = 0; h < q.heads; ++h)
                for (size_t qt = 0; qt < g.tiles(); ++qt) {
                    std::vector<long double> logits(g.tiles());
                    for (size_t kt = 0; kt < g.tiles(); ++kt) {
                        long double dot = 0;
                        for (size_t d = 0; d < q.dim; ++d)
                            dot += static_cast<long double>(qp.at(b, qt, h, d)) * kp.at(b, kt, h, d);
                        logits[kt] = dot / std::sqrt(static_cast<long double>(q.dim));
                        close(gated.routing.scores.at(b, h, qt, kt), static_cast<float>(logits[kt]),
                              "pooled score is not mean Q dot mean K / sqrt(D)");
                    }
                    long double maximum = *std::max_element(logits.begin(), logits.end()), total = 0;
                    for (long double& l : logits) { l = std::exp(l - maximum); total += l; }
                    for (size_t d = 0; d < q.dim; ++d) {
                        long double sum = 0;
                        for (size_t kt = 0; kt < g.tiles(); ++kt) sum += logits[kt] / total * vp.at(b, kt, h, d);
                        close(gated.compressed_tiles.at(b, qt, h, d), static_cast<float>(sum), "compressed weighted pooled V mismatch");
                    }
                }
        for (size_t b = 0; b < q.batch; ++b)
            for (size_t t = 0; t < q.tokens; ++t)
                for (size_t h = 0; h < q.heads; ++h)
                    for (size_t d = 0; d < q.dim; ++d) {
                        float c = gated.compressed_tiles.at(b, g.untile_combined_index[t] / elems, h, d);
                        close(gated.output.at(b, t, h, d), result.output.at(b, t, h, d) + gate.at(b, t, h, d) * c,
                              "gate fusion not additive/per-token/per-head/per-channel");
                    }
        AttentionResult dense_gated = attention(q, k, v, g, dense, &gate);
        tensor_close(dense_gated.fine, oracle, "dense step fine attention changed");
        tensor_close(dense_gated.compressed_tiles, gated.compressed_tiles, "dense mode disabled/changed compression", 0);
        sparse.force_dense = true;
        tensor_close(attention(q, k, v, g, sparse, &gate).output, dense_gated.output,
                     "dense schedule gate semantics differ from dense step", 0);
        check(q.data == q_before && k.data == k_before && v.data == v_before && gate.data == gate_before,
              "attention changed original Q/K/V/gate");
    }
    // Per-head REAL pooled Q/K routing, not just fabricated score-mask tests.
    Geometry g = make_geometry({Segment::text(1), Segment::video({1, 1, 8})}, 64);
    Tensor q(1, g.packed_tokens, 2, 1), k = q, v = q;
    for (size_t t = 0; t < q.tokens; ++t) {
        q.at(0, t, 0, 0) = 1; q.at(0, t, 1, 0) = -1;
        k.at(0, t, 0, 0) = k.at(0, t, 1, 0) = t < 5 ? 2.f : -2.f;
        v.at(0, t, 0, 0) = v.at(0, t, 1, 0) = static_cast<float>(t);
    }
    RoutingOptions sparse;
    auto result = attention(q, k, v, g, sparse);
    check(result.routing.keeps(0, 0, 1, 1) && !result.routing.keeps(0, 0, 1, 2) &&
          !result.routing.keeps(0, 1, 1, 1) && result.routing.keeps(0, 1, 1, 2), "actual score routing mixed heads");
    tensor_close(result.output, packed_oracle(q, k, v, g, &result.routing), "actual per-head sparse baseline");
    // Compression has one vote per tile, NOT proportional to tile token count.
    Geometry unequal = make_geometry({Segment::text(1), Segment::video({1, 1, 8})}, 64);
    Tensor uq(1, unequal.packed_tokens, 1, 1), uk = uq, uv = uq, gate = uq;
    for (size_t t = 0; t < uq.tokens; ++t) {
        uv.at(0, t, 0, 0) = t == 0 ? 1.f : (t < 5 ? 3.f : 9.f);
        gate.at(0, t, 0, 0) = t % 2 == 0 ? -2.f : 2.f; // negative and >1 are legal raw gates
    }
    auto gated = attention(uq, uk, uv, unequal, sparse, &gate);
    for (float c : gated.compressed_tiles.data) close(c, 13.f / 3, "compression counted padded tokens or selected-only keys");
    close(gated.fine.at(0, 1, 0, 0), 13.f / 5, "sparse prefix+selected token weighting wrong");
    close(gated.output.at(0, 1, 0, 0), 13.f / 5 + 2 * 13.f / 3, "gate improperly sigmoid/clipped/convex blended");
    // Prefix QUERY remains fine-dense, but active compression is STILL added.
    close(gated.fine.at(0, 0, 0, 0), 49.f / 9, "prefix dense-query baseline wrong");
    close(gated.output.at(0, 0, 0, 0), 49.f / 9 - 2 * 13.f / 3, "prefix gate was incorrectly disabled");
    Tensor hidden(1, 3, 1, 2); hidden.data = {1, 2, -1, 3, 0, 0};
    std::vector<float> weights{2, -1, -3, 1, 4, 0, 0, 2};
    const auto saved_hidden = hidden.data, saved_weights = weights;
    auto linear = linear_gate(hidden, weights, 2, 2);
    check(linear.data == std::vector<float>({0, -1, 4, 4, -5, 6, -4, 6, 0, 0, 0, 0}),
          "to_gate_compress must be bias-free raw linear, not sigmoid");
    check(hidden.data == saved_hidden && weights == saved_weights, "linear gate mutated hidden/weights");
    auto zero_linear = linear_gate(hidden, std::vector<float>(8, 0), 2, 2);
    check(std::all_of(zero_linear.data.begin(), zero_linear.data.end(), [](float x) { return x == 0; }),
          "zero trained gate incorrectly became sigmoid(0)=0.5");
}
void padding_and_invalid_tests() {
    for (size_t elems : {64u, 128u}) {
        Geometry g = make_geometry({Segment::text(1), Segment::video({1, 1, tile_shape(elems)[2] + 1})}, elems);
        check(g.tiles() % 2 == 1, "dummy fixture must have odd logical tiles");
        Tensor q = sample(1, g.packed_tokens, 2, 2, 1), k = sample(1, g.packed_tokens, 2, 2, 2),
               v = sample(1, g.packed_tokens, 2, 2, 3), gate = sample(1, g.packed_tokens, 2, 2, 4);
        RoutingOptions opts;
        auto ordinary = attention(q, k, v, g, opts, &gate);
        auto dummy = attention(q, k, v, g, opts, &gate, true);
        tensor_close(dummy.output, ordinary.output, "transport dummy affected attention", 0);
        check(dummy.routing.mask == ordinary.routing.mask && dummy.routing.scores.data == ordinary.routing.scores.data,
              "dummy counted in logical score search/selection");
        tensor_close(dummy.compressed_tiles, ordinary.compressed_tiles, "dummy affected compression", 0);
        Tensor tq = tile(q, g, true), tk = tile(k, g, true), tv = tile(v, g, true), tg = tile(gate, g, true);
        // Deliberately poison every INVALID slot. This CPU oracle ignores them;
        // upstream buffers guarantee zero padding before kernels, not NaN pads.
        for (size_t t = 0; t < tq.tokens; ++t)
            if (t >= g.padded_tokens() || g.source_index[t] == invalid_token)
                for (size_t h = 0; h < tq.heads; ++h)
                    for (size_t d = 0; d < tq.dim; ++d) {
                        tq.at(0, t, h, d) = std::numeric_limits<float>::quiet_NaN();
                        tk.at(0, t, h, d) = std::numeric_limits<float>::infinity();
                        tv.at(0, t, h, d) = -std::numeric_limits<float>::infinity();
                        tg.at(0, t, h, d) = std::numeric_limits<float>::quiet_NaN();
                    }
        Scores poisoned_scores = pooled_scores(tq, tk, g);
        check(poisoned_scores.data == ordinary.routing.scores.data, "padding poison leaked into pooled score");
        Routing routed = route(poisoned_scores, g, opts);
        Tensor fine = fine_attention(tq, tk, tv, g, routed), compressed = compressed_attention(tv, g, poisoned_scores);
        tensor_close(untile(fuse_compression(fine, compressed, tg, g), g), ordinary.output,
                     "padding poison leaked into fine/compression/gate output", 0);
        for (size_t t = 0; t < fine.tokens; ++t)
            if (t >= g.padded_tokens() || g.source_index[t] == invalid_token)
                for (size_t h = 0; h < fine.heads; ++h)
                    for (size_t d = 0; d < fine.dim; ++d)
                        check(fine.at(0, t, h, d) == 0, "invalid output query was computed");
        rejects([&] { Tensor bad(1, g.padded_tokens() + 2 * elems, 2, 2); untile(bad, g); }, "extra dummy tiles accepted");
        Tensor corrupt = q; corrupt.data.pop_back();
        rejects([&] { tile(corrupt, g); }, "short tensor storage accepted");
        corrupt = q; corrupt.data[0] = std::numeric_limits<float>::quiet_NaN();
        rejects([&] { attention(corrupt, k, v, g, opts); }, "nonfinite valid input accepted");
        corrupt = gate; corrupt.data[0] = std::numeric_limits<float>::infinity();
        rejects([&] { attention(q, k, v, g, opts, &corrupt); }, "nonfinite valid gate accepted");
        Tensor wrong(1, q.tokens, 1, 2);
        rejects([&] { attention(q, k, wrong, g, opts); }, "QKV shape mismatch accepted");
        rejects([&] { attention(q, k, v, g, opts, &wrong); }, "gate shape mismatch accepted");
        Tensor huge = q; std::fill(huge.data.begin(), huge.data.end(), std::numeric_limits<float>::max());
        rejects([&] { pooled_scores(tile(huge, g), tile(huge, g), g); }, "F32 pool/score overflow accepted");
        Routing empty = routed; std::fill(empty.mask.begin(), empty.mask.end(), uint8_t(0));
        rejects([&] { fine_attention(tq, tk, tv, g, empty); }, "empty selected key set accepted");
    }
    auto even = make_geometry({Segment::text(1), Segment::video({1, 1, 1})}, 64);
    Tensor input(1, even.packed_tokens, 1, 1);
    rejects([&] { tile(input, even, true); }, "dummy accepted even geometry");
    auto g256 = make_geometry({Segment::video({1, 1, 1})}, 256);
    Tensor one(1, 1, 1, 1);
    rejects([&] { tile(one, g256, true); }, "dummy accepted tile256");
    auto stable = softmax({10000, 10001, 9999});
    close(std::accumulate(stable.begin(), stable.end(), 0.f), 1, "softmax not max-subtracted");
    rejects([] { softmax({}); }, "empty softmax accepted");
    rejects([] { linear_gate(Tensor(1, 1, 1, 2), {1}, 1, 1); }, "bad gate weight shape accepted");
}
void capability_and_nonmutation_tests() {
    check(!native_cuda_implemented && !production_switch_available, "CPU oracle masquerading as native/production");
    auto missing = inspect_gate_weights({}, 2, 2, 2, 1);
    check(missing.capability == WeightCapability::Missing && !missing.production_ready, "base H3 accepted as VSA");
    std::vector<float> zero(4, 0), trained{0, 0.25f, 0, 0};
    std::vector<GateWeightView> views{
        {"transformer_blocks.0.attn.to_gate_compress.weight", 2, 2, &zero, nullptr},
        {"transformer_blocks.1.attn.to_gate_compress.weight", 2, 2, &zero, nullptr}};
    auto report = inspect_gate_weights(views, 2, 2, 2, 1);
    check(report.complete && report.capability == WeightCapability::ZeroOnly && !report.any_nonzero_gate &&
          !report.production_ready, "zero-initialized H3 VSA gate accepted");
    views[1].f32 = &trained;
    report = inspect_gate_weights(views, 2, 2, 2, 1);
    check(report.capability == WeightCapability::FloatReference && report.complete && report.any_nonzero_gate &&
          report.cpu_linear_gate_supported && !report.production_ready, "float gates should qualify only CPU linear reference");
    check(inspect_gate_weights(views, 3, 2, 2, 1).capability == WeightCapability::Missing, "incomplete layer coverage accepted");
    auto bad = views; bad[1].tensor_name = "transformer_blocks.1.ff.to_gate_compress.weight";
    check(inspect_gate_weights(bad, 2, 2, 2, 1).capability == WeightCapability::Invalid, "FFN tensor accepted as attention gate");
    bad = views; bad[1].output_dim = 3;
    check(inspect_gate_weights(bad, 2, 2, 2, 1).capability == WeightCapability::Invalid, "wrong architecture gate accepted");
    bad = views; bad[1].f32 = nullptr;
    check(inspect_gate_weights(bad, 2, 2, 2, 1).capability == WeightCapability::Missing, "missing trained gate accepted");
    std::vector<float> placeholder;
    bad = views; bad[1].f32 = &placeholder;
    check(inspect_gate_weights(bad, 2, 2, 2, 1).capability == WeightCapability::Missing, "offload placeholder accepted");
    auto nan = trained; nan[1] = std::numeric_limits<float>::quiet_NaN();
    bad = views; bad[1].f32 = &nan;
    check(inspect_gate_weights(bad, 2, 2, 2, 1).capability == WeightCapability::Invalid, "NaN gate weight accepted");
    auto short_weight = trained; short_weight.pop_back();
    bad = views; bad[1].f32 = &short_weight;
    check(inspect_gate_weights(bad, 2, 2, 2, 1).capability == WeightCapability::Invalid, "short gate storage accepted");
    std::vector<uint8_t> packed_zero{0x88, 0x00}, packed_active{0x88, 0x91};
    std::vector<GateWeightView> packed{
        {"transformer_blocks.0.attn.to_gate_compress.weight", 2, 2, nullptr, &packed_zero},
        {"transformer_blocks.1.attn.to_gate_compress.weight", 2, 2, nullptr, &packed_zero}};
    check(inspect_gate_weights(packed, 2, 2, 2, 1).capability == WeightCapability::ZeroOnly,
          "NVFP4 signed zeros treated as trained gates");
    packed[1].packed_nvfp4 = &packed_active;
    report = inspect_gate_weights(packed, 2, 2, 2, 1);
    check(report.capability == WeightCapability::PackedInspectionOnly && report.complete && report.any_nonzero_gate &&
          !report.cpu_linear_gate_supported && !report.production_ready, "packed inspection claimed execution without scales");
    auto short_packed = packed_active; short_packed.pop_back();
    packed[1].packed_nvfp4 = &short_packed;
    check(inspect_gate_weights(packed, 2, 2, 2, 1).capability == WeightCapability::Invalid, "bad NVFP4 storage accepted");
    // No FFN argument in any API; immutable model-sentinel state stays intact.
    struct DummyModel { std::vector<float> ffn_in, ffn_out, residual, gate_weight; };
    DummyModel model{{1, 2, 3, 4}, {-2, 6, 8}, {4, 5}, trained};
    DummyModel before = model;
    Geometry g = make_geometry({Segment::text(1), Segment::video({1, 1, 9})}, 64);
    Tensor q = sample(1, g.packed_tokens, 2, 1, 10), k = sample(1, g.packed_tokens, 2, 1, 11),
           v = sample(1, g.packed_tokens, 2, 1, 12), hidden = sample(1, g.packed_tokens, 1, 2, 13);
    Tensor raw = linear_gate(hidden, model.gate_weight, 2, 1);
    auto result = attention(q, k, v, g, RoutingOptions{}, &raw);
    check(result.output.tokens == g.packed_tokens && model.ffn_in == before.ffn_in && model.ffn_out == before.ffn_out &&
          model.residual == before.residual && model.gate_weight == before.gate_weight,
          "reference/capability guard mutated FFN/residual/model weights");
}
void randomized_tests() {
    std::mt19937 random(0x48335653);
    for (size_t run = 0; run < 24; ++run) {
        size_t elems = std::array<size_t, 3>{64, 128, 256}[run % 3];
        Grid a{1 + random() % 3, 1 + random() % 3, 5 + random() % 6};
        Grid b{1 + random() % 2, 1 + random() % 3, 5 + random() % 6};
        Geometry g = make_geometry({Segment::audio(random() % 4), Segment::video(a), Segment::text(random() % 4),
                                   Segment::reference(random() % 4), Segment::video(b)}, elems);
        check(g.tile_partition_indices == independent_partition(g.segments, elems), "random geometry tile order");
        size_t batch = 1 + run % 2, heads = 1 + run % 3, dim = 1 + run % 4;
        Tensor q = sample(batch, g.packed_tokens, heads, dim, uint32_t(run * 3)),
               k = sample(batch, g.packed_tokens, heads, dim, uint32_t(run * 3 + 1)),
               v = sample(batch, g.packed_tokens, heads, dim, uint32_t(run * 3 + 2));
        RoutingOptions opts; opts.sparsity = 0.7; opts.reference_keep_rate = 0.2;
        auto sparse = attention(q, k, v, g, opts);
        tensor_close(sparse.output, packed_oracle(q, k, v, g, &sparse.routing), "random masked-attention oracle");
        opts.force_dense = true;
        tensor_close(attention(q, k, v, g, opts).output, packed_oracle(q, k, v, g), "random dense-attention oracle");
    }
}
// Optional machine-readable fixture for independent, source-extracted NumPy
// CPU comparison. Test binary has no Python/torch/runtime dependency.
template<class T> void json_vector(const std::vector<T>& values) {
    std::cout << '[';
    for (size_t i = 0; i < values.size(); ++i) { if (i) std::cout << ','; std::cout << +values[i]; }
    std::cout << ']';
}
void dump_oracle() {
    std::cout << std::setprecision(9) << '[';
    for (size_t elems : {64u, 128u, 256u}) {
        if (elems != 64) std::cout << ',';
        Geometry g = make_geometry({Segment::text(3), Segment::video({2, 3, 9}), Segment::audio(2),
                                   Segment::reference(1), Segment::video({1, 2, 9})}, elems);
        Tensor q(2, g.packed_tokens, 2, 3), k = q, v = q, gate = q;
        for (size_t i = 0; i < q.data.size(); ++i) {
            q.data[i] = static_cast<float>(static_cast<int>((i * 13 + 7) % 97) - 48) / 64 + 1;
            k.data[i] = static_cast<float>(static_cast<int>((i * 17 + 3) % 89) - 44) / 64 + static_cast<float>(g.untile_combined_index[(i / 6) % q.tokens] / elems) / 2;
            v.data[i] = static_cast<float>(static_cast<int>((i * 19 + 11) % 83) - 41) / 32;
            gate.data[i] = static_cast<float>(static_cast<int>((i * 23 + 5) % 71) - 35) / 16;
        }
        RoutingOptions opts; opts.sparsity = 0.8; opts.reference_keep_rate = 0.3;
        auto result = attention(q, k, v, g, opts, &gate);
        std::cout << "{\"tile_elems\":" << elems << ",\"tokens\":" << q.tokens << ",\"prefix\":" << g.num_prefix_tiles;
        std::cout << ",\"sizes\":"; json_vector(g.variable_block_sizes);
        std::cout << ",\"partition\":"; json_vector(g.tile_partition_indices);
        std::cout << ",\"inverse\":"; json_vector(g.untile_combined_index);
        std::cout << ",\"q\":"; json_vector(q.data); std::cout << ",\"k\":"; json_vector(k.data);
        std::cout << ",\"v\":"; json_vector(v.data); std::cout << ",\"gate\":"; json_vector(gate.data);
        std::cout << ",\"scores\":"; json_vector(result.routing.scores.data);
        std::cout << ",\"mask\":"; json_vector(result.routing.mask);
        std::cout << ",\"fine\":"; json_vector(result.fine.data);
        std::cout << ",\"compressed\":"; json_vector(result.compressed_tiles.data);
        std::cout << ",\"output\":"; json_vector(result.output.data); std::cout << '}';
    }
    std::cout << "]\n";
}
} // namespace
int main(int argc, char** argv) {
    try {
        if (argc == 2 && std::string(argv[1]) == "--dump-oracle") { dump_oracle(); return 0; }
        geometry_tests(); routing_tests(); attention_tests(); padding_and_invalid_tests();
        capability_and_nonmutation_tests(); randomized_tests();
        std::cout << "H3 VSA CPU reference: " << checks << " checks passed; no GPU/native/production activation\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "H3 VSA CPU reference FAILED after " << checks << " checks: " << e.what() << '\n';
        return 1;
    }
}
