#include <cstdio>
#include <cstdint>
#include "ggml.h"

extern "C" void lightdit_ggml_smoke_test() {
    ggml_init_params params;
    params.mem_size   = 16 * 1024 * 1024;
    params.mem_buffer = nullptr;
    params.no_alloc   = false;

    ggml_context* ctx = ggml_init(params);
    if (!ctx) {
        std::fprintf(stderr, "ggml_init failed\n");
        return;
    }

    ggml_tensor* a = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 4);
    if (!a || !a->data) {
        std::fprintf(stderr, "ggml_new_tensor_1d failed\n");
        ggml_free(ctx);
        return;
    }

    float* data = static_cast<float*>(a->data);
    for (int i = 0; i < 4; ++i) {
        data[i] = static_cast<float>(i + 1);
    }

    std::printf("ggml smoke tensor: [%f, %f, %f, %f]\n",
                data[0], data[1], data[2], data[3]);

    std::printf("ggml version: %s\n", ggml_version());
    std::printf("ggml commit: %s\n", ggml_commit());

    ggml_free(ctx);
}