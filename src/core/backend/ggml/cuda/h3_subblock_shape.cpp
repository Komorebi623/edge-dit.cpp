#include "h3_subblock.h"

namespace {
using namespace edgedit::h3_subblock;
bool matrix(const ggml_tensor* t, ggml_type type, int64_t n0, int64_t n1) {
    return t && t->type == type && t->ne[0] == n0 && t->ne[1] == n1 &&
           t->ne[2] == 1 && t->ne[3] == 1 && ggml_is_contiguous(t);
}
bool qshape(const ggml_tensor* q) {
    if (!q || q->type != GGML_TYPE_F32 || q->ne[0] != dimension || q->ne[1] < 1 ||
        q->ne[1] > max_sequence || q->ne[2] < 1 || q->ne[2] > max_heads || q->ne[3] != 1 ||
        q->nb[0] != sizeof(float)) return false;
    for (int i = 1; i < 4; ++i)
        if (!q->nb[i] || q->nb[i] % sizeof(float) || q->nb[i] > (size_t(1) << 40)) return false;
    return true;
}
}

bool ed_cuda_h3_subblock_shape_supported(const ggml_tensor* dst) {
    using namespace edgedit::h3_subblock;
    const auto magic = tag(dst);
    if (magic != router_tag && magic != attention_tag && magic != indexed_attention_tag) return false;
    const auto* q = dst->src[0]; const auto* k = dst->src[1];
    if (!qshape(q) || !qshape(k) || k->ne[1] != q->ne[1] || k->ne[2] != q->ne[2] ||
        !ggml_is_contiguous(dst)) return false;
    if (magic == router_tag) {
        const auto* blocks = dst->src[2];
        if (!blocks || blocks->ne[1] < 1 || blocks->ne[1] > max_blocks ||
            !matrix(blocks, GGML_TYPE_I32, 2, blocks->ne[1]) ||
            dst->type != GGML_TYPE_I32 || dst->ne[1] != blocks->ne[1] ||
            dst->ne[2] != q->ne[2] || dst->ne[3] != 1 ||
            dst->ne[0] < 1 || dst->ne[0] > blocks->ne[1]) return false;
        const auto b = blocks->ne[1], h = q->ne[2];
        return size_t(h*b*4*dimension)*sizeof(float)*2 + size_t(h*b*b)*sizeof(float) <= router_limit;
    }
    const auto* v = dst->src[2]; const auto* route = dst->src[3];
    const auto* blocks = dst->src[4]; const auto* chunks = dst->src[5]; const auto* ids = dst->src[6];
    if (!v || v->type != GGML_TYPE_F32 || v->ne[0] != dimension || v->ne[1] != q->ne[2] ||
        v->ne[2] != q->ne[1] || v->ne[3] != 1 || v->nb[0] != sizeof(float)) return false;
    for (int i = 1; i < 4; ++i)
        if (!v->nb[i] || v->nb[i] % sizeof(float) || v->nb[i] > (size_t(1) << 40)) return false;
    if (!route || !blocks || !chunks || !ids ||
        route->type != GGML_TYPE_I32 || !ggml_is_contiguous(route) ||
        route->ne[0] < 1 || route->ne[0] > blocks->ne[1] ||
        route->ne[1] != blocks->ne[1] || route->ne[2] != q->ne[2] || route->ne[3] != 1 ||
        !matrix(blocks, GGML_TYPE_I32, 2, blocks->ne[1]) ||
        !matrix(chunks, GGML_TYPE_I32, 2, chunks->ne[1]) || chunks->ne[1] < 1 ||
        chunks->ne[1] > q->ne[1] + 1 ||
        !matrix(ids, GGML_TYPE_I32, ids->ne[0], 1) || ids->ne[0] < 1 || ids->ne[0] > q->ne[1] ||
        dst->type != GGML_TYPE_F32 || dst->ne[0] != dimension*q->ne[2] ||
        dst->ne[1] != q->ne[1] || dst->ne[2] != 1 || dst->ne[3] != 1) return false;
    Capacity c;
    return capacity(q->ne[1], q->ne[2], blocks->ne[1], route->ne[0], ids->ne[0]-1, &c);
}
