#pragma once

#include <cstdint>

namespace lightdit {

inline float calculate_shift(int64_t image_seq_len,
                             int64_t base_seq_len = 256,
                             int64_t max_seq_len  = 4096,
                             float base_shift      = 0.5f,
                             float max_shift       = 1.15f) {
    float m = (max_shift - base_shift) / static_cast<float>(max_seq_len - base_seq_len);
    float b = base_shift - m * static_cast<float>(base_seq_len);
    return static_cast<float>(image_seq_len) * m + b;
}

inline int64_t packed_latent_extent(int64_t image_extent, int64_t vae_scale_factor) {
    return 2 * (image_extent / (vae_scale_factor * 2));
}

inline int64_t packed_latent_seq_len(int64_t width, int64_t height, int64_t vae_scale_factor) {
    int64_t latent_w = packed_latent_extent(width, vae_scale_factor);
    int64_t latent_h = packed_latent_extent(height, vae_scale_factor);
    return (latent_w / 2) * (latent_h / 2);
}

}  // namespace lightdit
