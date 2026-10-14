#pragma once

#include "backend/ggml/ggml_extend.hpp"

#include <cmath>
#include <limits>
#include <stdexcept>

namespace MiniMaxH3 {

// H3-only immutable dynamic LoRA. Construct in the runner's unallocated
// params_ctx, then allocate the runner and load tensors() using a
// separate ModelLoader initialized with init_from_file (no name conversion).
// A=[in,rank], B=[rank,out], in ggml order. Neither factor is converted/scaled:
// coefficient = user multiplier * alpha/rank (or just multiplier if the input
// was already normalized numerically). No implicit alpha/rank is applied.
// The runner owns all tensors/buffers; this object only binds their identities.
class DynamicLoraAdapter final : public WeightAdapter {
public:
    using TensorMap = ModelLoader::TensorMap;
    struct Coverage {
        size_t target_count = 0;
        std::vector<std::string> used;    // Full native base weight names.
        std::vector<std::string> unused;
        bool complete() const { return unused.empty(); }
    };

    DynamicLoraAdapter(ggml_context* params_ctx,
                       const TensorMap& base_tensor_map,
                       const String2TensorStorage& adapter_storage_map,
                       float coefficient = 1.f)
        : coefficient_(coefficient) {
        require(params_ctx && ggml_get_no_alloc(params_ctx), "unallocated runner params_ctx required");
        require(std::isfinite(coefficient), "nonfinite coefficient");
        require(!adapter_storage_map.empty(), "empty adapter");
        std::set<std::string> existing_names;
        std::set<ggml_tensor*> existing_tensors;
        for (auto* t = ggml_get_first_tensor(params_ctx); t; t = ggml_get_next_tensor(params_ctx, t)) {
            require(!t->data && !t->buffer, "construct before runner allocation");
            existing_names.insert(ggml_get_name(t));
            existing_tensors.insert(t);
        }
        struct Pending { const TensorStorage* a = nullptr; const TensorStorage* b = nullptr; };
        std::map<std::string, Pending> pending;
        // Validate ALL metadata before adding any tensors to the runner context.
        // A map cannot expose duplicate file keys already overwritten by a loader;
        // reject those in the converter/parser, before constructing this map.
        for (const auto& entry : adapter_storage_map) {
            const auto& key = entry.first;
            const auto& s = entry.second;
            require(key == s.name, "storage key/name mismatch or duplicate alias: " + key);
            const bool is_a = ends_with(key, ".lora_A.weight");
            require(is_a || ends_with(key, ".lora_B.weight"), "unsupported key: " + key);
            const std::string module = key.substr(0, key.size() - std::string(".lora_A.weight").size());
            const std::string root = "model.diffusion_model.";
            require(module.compare(0, root.size(), root) == 0 && module.size() > root.size() &&
                    module.back() != '.' && module.find("..") == std::string::npos &&
                    module.find_first_not_of("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.") == std::string::npos,
                    "non-native H3 target: " + key);
            require(key.size() < std::min<size_t>(128, GGML_MAX_NAME), "tensor name would be truncated: " + key);
            require(!existing_names.count(key) && !base_tensor_map.count(key), "duplicate tensor name: " + key);
            require(s.type == GGML_TYPE_F32 || s.type == GGML_TYPE_F16 || s.type == GGML_TYPE_BF16,
                    "factor must be F32/F16/BF16: " + key);
            require((s.expected_type == GGML_TYPE_COUNT || s.expected_type == s.type) &&
                    !s.is_f8_e4m3 && !s.is_f8_e5m2 && !s.is_f64 && !s.is_i64 &&
                    !s.swap_scale_shift && !s.swap_swiglu_halves, "non-normalized storage: " + key);
            require(s.n_dims == 2 && s.ne[0] > 0 && s.ne[1] > 0, "factor must be a positive matrix: " + key);
            for (int d = 2; d < ED_MAX_DIMS; ++d) require(s.ne[d] == 1, "extra factor dimension: " + key);
            const uint64_t limit = std::min<uint64_t>(std::numeric_limits<int64_t>::max(),
                                                       std::numeric_limits<size_t>::max());
            require(uint64_t(s.ne[0]) <= limit / ggml_type_size(s.type) / uint64_t(s.ne[1]),
                    "factor size overflow: " + key);
            auto& p = pending[module + ".weight"];
            auto*& slot = is_a ? p.a : p.b;
            require(!slot, "duplicate factor: " + key);
            slot = &s;
        }
        for (const auto& entry : pending) {
            const auto& name = entry.first;
            const auto& p = entry.second;
            require(p.a && p.b, "incomplete A/B pair: " + name);
            const auto it = base_tensor_map.find(name);
            require(it != base_tensor_map.end() && it->second, "unknown base target: " + name);
            auto* w = it->second;
            require(existing_tensors.count(w) && w->op == GGML_OP_NONE && !w->view_src && ggml_is_contiguous(w),
                    "target must be an original runner parameter: " + name);
            require(w->ne[0] > 0 && w->ne[1] > 0 && w->ne[2] == 1 && w->ne[3] == 1 &&
                    (w->type == GGML_TYPE_F32 || w->type == GGML_TYPE_F16 || w->type == GGML_TYPE_BF16 ||
                     ggml_is_quantized(w->type)), "base target is not a linear weight: " + name);
            require(p.a->ne[1] <= 2048 && p.a->ne[1] == p.b->ne[0] &&
                    p.a->ne[0] == w->ne[0] && p.b->ne[1] == w->ne[1], "A/B/base shape or rank mismatch: " + name);
            require(base_names_.emplace(w, name).second, "duplicate base binding: " + name);
        }
        require(adapter_storage_map.size() <=
                (ggml_get_mem_size(params_ctx) - ggml_used_mem(params_ctx)) / ggml_tensor_overhead(),
                "insufficient runner parameter metadata capacity");
        for (const auto& entry : pending) {
            const auto make = [&](const TensorStorage& s) {
                auto* t = ggml_new_tensor_2d(params_ctx, s.type, s.ne[0], s.ne[1]);
                ggml_set_name(t, s.name.c_str());
                // Inference weights must remain graph leaves. ggml_set_param is
                // a TRAINING flag: it promotes op-NONE tensors into graph nodes,
                // hiding them from graph-cut INPUT_PARAM staging/accounting.
                // Runner params_ctx + WEIGHTS buffer provide ownership instead.
                tensors_.emplace(s.name, t);
                return t;
            };
            pairs_.emplace(entry.first, Pair{base_tensor_map.at(entry.first), make(*entry.second.a), make(*entry.second.b)});
        }
    }

    // Mutable only for the existing ModelLoader API; do not replace bindings.
    TensorMap& tensors() { return tensors_; }
    const TensorMap& tensors() const { return tensors_; }

    void begin_graph() {
        used_.clear();
        graph_open_ = true;
    }

    ggml_tensor* patch_weight(ggml_context*, ggml_backend_t, ggml_tensor* weight,
                              const std::string& weight_name) override {
        require(!bound_pair(weight, weight_name), "LoRA target used outside linear forward: " + weight_name);
        return weight; // Includes unrelated biases, norms, convolutions, and weights.
    }

    ggml_tensor* forward_with_lora(ggml_context* ctx, ggml_backend_t,
                                   ggml_tensor* x, ggml_tensor* w, ggml_tensor* b,
                                   const std::string& prefix, ForwardParams params) override {
        const std::string name = prefix + "weight"; // EXACT baseline Linear name.
        const Pair* p = bound_pair(w, name);
        if (params.op_type != ForwardParams::op_type_t::OP_LINEAR) {
            require(!p, "LoRA target used by a non-linear operation: " + name);
            require(params.op_type == ForwardParams::op_type_t::OP_CONV2D, "unsupported forward operation");
            const auto& c = params.conv2d;
            return ggml_ext_conv_2d(ctx, x, w, b, c.s0, c.s1, c.p0, c.p1, c.d0, c.d1,
                                    c.direct, c.circular_x, c.circular_y, c.scale);
        }
        if (p) {
            require(graph_open_, "begin_graph required before target forward");
            require(x && x->ne[0] == w->ne[0], "activation/weight shape mismatch: " + name);
            for (int d = 0; d < GGML_MAX_DIMS; ++d) require(x->ne[d] > 0, "nonpositive activation dimension");
        }
        auto* out = ggml_ext_linear(ctx, x, w, b, params.linear.force_prec_f32, params.linear.scale, name.c_str());
        if (!p) return out;
        if (coefficient_ == 0.f) {
            used_.insert(name);
            return out; // No adapter nodes, casts, or renames.
        }
        // Base scale is conditioning (scale x, matmul, unscale), NOT LoRA gain.
        auto* lora_x = x->type == GGML_TYPE_F32 ? x : ggml_cast(ctx, x, GGML_TYPE_F32);
        // The base conditioning scale may have made its own contiguous copy;
        // this unscaled branch must also support the helper's large-batch reshape.
        if (lora_x->ne[2] * lora_x->ne[3] > 1024 && !ggml_is_contiguous(lora_x)) {
            lora_x = ggml_cont(ctx, lora_x);
        }
        auto* delta = ggml_ext_linear(ctx, lora_x, p->a, nullptr, params.linear.force_prec_f32, 1.f);
        delta = ggml_ext_linear(ctx, delta, p->b, nullptr, params.linear.force_prec_f32, 1.f);
        if (coefficient_ != 1.f) delta = ggml_scale(ctx, delta, coefficient_);
        auto* result = ggml_add(ctx, out, delta);
        used_.insert(name);
        return result;
    }

    // With a graph, inspect actual reachability (before execution/splitting),
    // including branches built but discarded. Without one, report only calls
    // since begin_graph(). No state is shared between instances or graphs.
    Coverage report(const ggml_cgraph* graph = nullptr) const {
        if (!graph) {
            Coverage result;
            result.target_count = pairs_.size();
            for (const auto& entry : pairs_) {
                (used_.count(entry.first) ? result.used : result.unused).push_back(entry.first);
            }
            return result;
        }
        std::set<const ggml_tensor*> weights;
        std::set<std::pair<const ggml_tensor*, const ggml_tensor*>> chains;
        // ggml's graph accessors lack const overloads; they do not mutate graph.
        auto* g = const_cast<ggml_cgraph*>(graph);
        for (int i = 0; i < ggml_graph_n_nodes(g); ++i) {
            const auto* node = ggml_graph_node(g, i);
            if (node->op != GGML_OP_MUL_MAT) continue;
            weights.insert(node->src[0]);
            const auto* input = node->src[1];
            while (input && input->op == GGML_OP_RESHAPE) input = input->src[0];
            if (input && input->op == GGML_OP_MUL_MAT) chains.emplace(input->src[0], node->src[0]);
        }
        Coverage result;
        result.target_count = pairs_.size();
        for (const auto& entry : pairs_) {
            const auto& p = entry.second;
            const bool used = used_.count(entry.first) && weights.count(p.base) &&
                              (coefficient_ == 0.f || chains.count({p.a, p.b}));
            (used ? result.used : result.unused).push_back(entry.first);
        }
        return result;
    }

    void validate_coverage(const ggml_cgraph* graph = nullptr) {
        require(graph_open_, "begin_graph required before coverage validation");
        const auto coverage = report(graph);
        if (!coverage.complete()) {
            std::string message = "targets absent from full graph:";
            for (const auto& name : coverage.unused) message += " " + name;
            require(false, message);
        }
        graph_open_ = false;
    }

    size_t get_extra_graph_size() override { return coefficient_ == 0.f ? 0 : pairs_.size() * 16; }

private:
    struct Pair { ggml_tensor* base; ggml_tensor* a; ggml_tensor* b; };
    const float coefficient_;
    TensorMap tensors_;
    std::map<std::string, Pair> pairs_;
    std::map<ggml_tensor*, std::string> base_names_;
    std::set<std::string> used_;
    bool graph_open_ = false;

    static void require(bool ok, const std::string& message) {
        if (!ok) throw std::invalid_argument("H3 dynamic LoRA: " + message);
    }
    static bool ends_with(const std::string& value, const std::string& suffix) {
        return value.size() >= suffix.size() && value.compare(value.size() - suffix.size(), suffix.size(), suffix) == 0;
    }
    const Pair* bound_pair(ggml_tensor* w, const std::string& name) const {
        const auto it = pairs_.find(name);
        if (it != pairs_.end()) {
            require(w == it->second.base, "base identity changed: " + name);
            return &it->second;
        }
        require(!base_names_.count(w), "base target called under a different name: " + name);
        return nullptr;
    }
};

} // namespace MiniMaxH3
