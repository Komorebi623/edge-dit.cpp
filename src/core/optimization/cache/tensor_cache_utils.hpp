#pragma once

#include <cmath>
#include <vector>

#include "utils/tensor.hpp"

namespace lightdit {
namespace cache {

inline bool store_tensor_diff(std::vector<float>* diff,
                              const sd::Tensor<float>& input,
                              const sd::Tensor<float>& output) {
    if (diff == nullptr || input.empty() || output.empty()) {
        return false;
    }

    const size_t input_size = static_cast<size_t>(input.numel());
    const size_t output_size = static_cast<size_t>(output.numel());
    if (input_size == 0 || input_size != output_size) {
        diff->clear();
        return false;
    }

    const float* input_data = input.data();
    const float* output_data = output.data();
    if (input_data == nullptr || output_data == nullptr) {
        diff->clear();
        return false;
    }

    diff->resize(output_size);
    for (size_t i = 0; i < output_size; ++i) {
        (*diff)[i] = output_data[i] - input_data[i];
    }
    return true;
}

inline bool apply_tensor_diff(const std::vector<float>& diff,
                              const sd::Tensor<float>& input,
                              sd::Tensor<float>* output) {
    if (output == nullptr || input.empty() || diff.empty()) {
        return false;
    }

    const size_t input_size = static_cast<size_t>(input.numel());
    if (input_size == 0 || diff.size() != input_size) {
        return false;
    }

    *output = input;
    float* output_data = output->data();
    if (output_data == nullptr) {
        return false;
    }

    for (size_t i = 0; i < input_size; ++i) {
        output_data[i] += diff[i];
    }
    return true;
}

inline float residual_diff(const float* prev, const float* curr, size_t size) {
    if (prev == nullptr || curr == nullptr || size == 0) {
        return 0.0f;
    }

    float sum_diff = 0.0f;
    float sum_abs = 0.0f;
    for (size_t i = 0; i < size; ++i) {
        sum_diff += std::fabs(prev[i] - curr[i]);
        sum_abs += std::fabs(prev[i]);
    }
    return sum_diff / (sum_abs + 1e-6f);
}

inline float residual_diff(const std::vector<float>& prev, const std::vector<float>& curr) {
    if (prev.empty() || prev.size() != curr.size()) {
        return 1.0f;
    }
    return residual_diff(prev.data(), curr.data(), prev.size());
}

}  // namespace cache
}  // namespace lightdit
