#include "utils/util.h"

#include <algorithm>
#include <cmath>
#include <codecvt>
#include <cstdarg>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <locale>
#include <sstream>
#include <thread>

#include "utils/tensor.hpp"

bool ends_with(const std::string& str, const std::string& ending) {
    return str.size() >= ending.size() &&
           str.compare(str.size() - ending.size(), ending.size(), ending) == 0;
}

bool starts_with(const std::string& str, const std::string& start) {
    return str.size() >= start.size() && str.compare(0, start.size(), start) == 0;
}

bool contains(const std::string& str, const std::string& substr) {
    return str.find(substr) != std::string::npos;
}

std::string sd_format(const char* fmt, ...) {
    va_list args;
    va_start(args, fmt);
    va_list args_copy;
    va_copy(args_copy, args);
    const int needed = std::vsnprintf(nullptr, 0, fmt, args_copy);
    va_end(args_copy);

    std::string result;
    if (needed > 0) {
        result.resize(static_cast<size_t>(needed));
        std::vsnprintf(result.data(), result.size() + 1, fmt, args);
    }
    va_end(args);
    return result;
}

std::u32string utf8_to_utf32(const std::string& utf8_str) {
    std::wstring_convert<std::codecvt_utf8<char32_t>, char32_t> converter;
    return converter.from_bytes(utf8_str);
}

std::string utf32_to_utf8(const std::u32string& utf32_str) {
    std::wstring_convert<std::codecvt_utf8<char32_t>, char32_t> converter;
    return converter.to_bytes(utf32_str);
}

std::u32string unicode_value_to_utf32(int unicode_value) {
    return {static_cast<char32_t>(unicode_value)};
}

std::vector<std::pair<std::string, float>> parse_prompt_attention(const std::string& text) {
    // Minimal parser for the first Flux path. Weight syntax can be expanded later;
    // returning one segment keeps the tokenizer contract identical.
    if (text.empty()) {
        return {};
    }
    return {{text, 1.0f}};
}

namespace {

double clip_preprocess_cubic_filter(double x) {
    constexpr double a = -0.5;
    x                  = std::abs(x);
    if (x < 1.0) {
        return ((a + 2.0) * x - (a + 3.0)) * x * x + 1.0;
    }
    if (x < 2.0) {
        return (((x - 5.0) * x + 8.0) * x - 4.0) * a;
    }
    return 0.0;
}

struct ClipPreprocessResizeAxisContrib {
    int start = 0;
    std::vector<int16_t> coeffs;
};

struct ClipPreprocessResizeAxis {
    std::vector<ClipPreprocessResizeAxisContrib> contribs;
    int precision_bits = 0;
};

ClipPreprocessResizeAxis clip_preprocess_precompute_bicubic_coeffs(int in_size,
                                                                   int out_size) {
    const double scale = static_cast<double>(in_size) / static_cast<double>(out_size);
    const double filterscale = std::max(1.0, scale);
    const double support = 2.0 * filterscale;
    const int ksize = static_cast<int>(std::ceil(support)) * 2 + 1;
    const double ss = 1.0 / filterscale;

    ClipPreprocessResizeAxis result;
    result.contribs.resize(static_cast<size_t>(out_size));
    std::vector<std::vector<double>> weights_by_output(static_cast<size_t>(out_size));
    double weight_max = 0.0;

    for (int out = 0; out < out_size; ++out) {
        const double center = (static_cast<double>(out) + 0.5) * scale;
        int xmin = static_cast<int>(center - support + 0.5);
        xmin = std::max(0, xmin);
        int xmax = static_cast<int>(center + support + 0.5);
        xmax = std::min(in_size, xmax);
        const int count = std::max(0, xmax - xmin);

        std::vector<double> weights(static_cast<size_t>(ksize), 0.0);
        double weight_sum = 0.0;
        for (int i = 0; i < count; ++i) {
            const double weight = clip_preprocess_cubic_filter((static_cast<double>(i + xmin) - center + 0.5) * ss);
            weights[static_cast<size_t>(i)] = weight;
            weight_sum += weight;
        }
        if (weight_sum != 0.0) {
            for (int i = 0; i < count; ++i) {
                weights[static_cast<size_t>(i)] /= weight_sum;
            }
        }

        for (int i = 0; i < count; ++i) {
            weight_max = std::max(weight_max, weights[static_cast<size_t>(i)]);
        }
        weights.resize(static_cast<size_t>(count));

        auto& axis = result.contribs[static_cast<size_t>(out)];
        axis.start = xmin;
        weights_by_output[static_cast<size_t>(out)] = std::move(weights);
    }

    while (result.precision_bits < 22) {
        const int next_value = static_cast<int>(0.5 + weight_max * static_cast<double>(1 << (result.precision_bits + 1)));
        if (next_value >= (1 << 15)) {
            break;
        }
        ++result.precision_bits;
    }

    const double fixed_scale = static_cast<double>(1 << result.precision_bits);
    for (int out = 0; out < out_size; ++out) {
        auto& axis = result.contribs[static_cast<size_t>(out)];
        const auto& weights = weights_by_output[static_cast<size_t>(out)];
        axis.coeffs.resize(weights.size());
        for (size_t i = 0; i < weights.size(); ++i) {
            const double scaled = weights[i] * fixed_scale;
            axis.coeffs[i] = static_cast<int16_t>(scaled < 0.0 ? scaled - 0.5 : scaled + 0.5);
        }
    }
    return result;
}

uint8_t clip_preprocess_to_u8(float value) {
    if (value <= 0.0f) {
        return 0;
    }
    if (value >= 1.0f) {
        return 255;
    }
    return static_cast<uint8_t>(value * 255.0f + 0.5f);
}

uint8_t clip_preprocess_fixed_clip8(int64_t value, int precision_bits) {
    const int64_t shifted = value >> precision_bits;
    if (shifted <= 0) {
        return 0;
    }
    if (shifted >= 255) {
        return 255;
    }
    return static_cast<uint8_t>(shifted);
}

sd::Tensor<float> clip_preprocess_resize_bicubic_u8(const sd::Tensor<float>& image,
                                                    int target_width,
                                                    int target_height) {
    const int src_width = static_cast<int>(image.shape()[0]);
    const int src_height = static_cast<int>(image.shape()[1]);
    const int src_channels = static_cast<int>(image.shape()[2]);
    const int frames = image.dim() >= 4 ? static_cast<int>(image.shape()[3]) : 1;
    const auto x_axis = clip_preprocess_precompute_bicubic_coeffs(src_width, target_width);
    const auto y_axis = clip_preprocess_precompute_bicubic_coeffs(src_height, target_height);

    sd::Tensor<float> output({target_width, target_height, 3, frames});
    std::vector<uint8_t> src_u8(static_cast<size_t>(src_width) * static_cast<size_t>(src_height) * 3);
    std::vector<uint8_t> tmp(static_cast<size_t>(src_height) * static_cast<size_t>(target_width) * 3);

    for (int f = 0; f < frames; ++f) {
        for (int y = 0; y < src_height; ++y) {
            for (int x = 0; x < src_width; ++x) {
                uint8_t* pixel = src_u8.data() +
                                 (static_cast<size_t>(y) * static_cast<size_t>(src_width) +
                                  static_cast<size_t>(x)) *
                                     3;
                for (int c = 0; c < 3; ++c) {
                    const int src_c = c < src_channels ? c : 0;
                    pixel[c] = clip_preprocess_to_u8(image.index(x, y, src_c, f));
                }
            }
        }

        for (int y = 0; y < src_height; ++y) {
            for (int x = 0; x < target_width; ++x) {
                const auto& coeff = x_axis.contribs[static_cast<size_t>(x)];
                int64_t accum[3] = {
                    1LL << (x_axis.precision_bits - 1),
                    1LL << (x_axis.precision_bits - 1),
                    1LL << (x_axis.precision_bits - 1),
                };
                for (size_t k = 0; k < coeff.coeffs.size(); ++k) {
                    const int src_x = coeff.start + static_cast<int>(k);
                    const uint8_t* pixel = src_u8.data() +
                                           (static_cast<size_t>(y) * static_cast<size_t>(src_width) +
                                            static_cast<size_t>(src_x)) *
                                               3;
                    for (int c = 0; c < 3; ++c) {
                        accum[c] += static_cast<int64_t>(pixel[c]) * static_cast<int64_t>(coeff.coeffs[k]);
                    }
                }

                uint8_t* out = tmp.data() +
                               (static_cast<size_t>(y) * static_cast<size_t>(target_width) +
                                   static_cast<size_t>(x)) *
                                   3;
                out[0] = clip_preprocess_fixed_clip8(accum[0], x_axis.precision_bits);
                out[1] = clip_preprocess_fixed_clip8(accum[1], x_axis.precision_bits);
                out[2] = clip_preprocess_fixed_clip8(accum[2], x_axis.precision_bits);
            }
        }

        for (int y = 0; y < target_height; ++y) {
            const auto& coeff = y_axis.contribs[static_cast<size_t>(y)];
            for (int x = 0; x < target_width; ++x) {
                int64_t accum[3] = {
                    1LL << (y_axis.precision_bits - 1),
                    1LL << (y_axis.precision_bits - 1),
                    1LL << (y_axis.precision_bits - 1),
                };
                for (size_t k = 0; k < coeff.coeffs.size(); ++k) {
                    const int src_y = coeff.start + static_cast<int>(k);
                    const uint8_t* pixel = tmp.data() +
                                           (static_cast<size_t>(src_y) * static_cast<size_t>(target_width) +
                                            static_cast<size_t>(x)) *
                                               3;
                    for (int c = 0; c < 3; ++c) {
                        accum[c] += static_cast<int64_t>(pixel[c]) * static_cast<int64_t>(coeff.coeffs[k]);
                    }
                }
                for (int c = 0; c < 3; ++c) {
                    output.index(x, y, c, f) =
                        static_cast<float>(clip_preprocess_fixed_clip8(accum[c], y_axis.precision_bits)) / 255.0f;
                }
            }
        }
    }

    return output;
}

}  // namespace

sd::Tensor<float> clip_preprocess(const sd::Tensor<float>& image, int target_width, int target_height) {
    if (image.empty() || target_width <= 0 || target_height <= 0 || image.dim() < 3) {
        return {};
    }

    const int64_t src_c = image.shape()[2];
    const int64_t frames = image.dim() >= 4 ? image.shape()[3] : 1;
    if (src_c < 3) {
        return {};
    }

    sd::Tensor<float> resized = image;
    if (image.shape()[0] != target_width || image.shape()[1] != target_height) {
        resized = clip_preprocess_resize_bicubic_u8(image, target_width, target_height);
    }

    static constexpr float mean[3] = {0.48145466f, 0.45782750f, 0.40821073f};
    static constexpr float std[3]  = {0.26862954f, 0.26130258f, 0.27577711f};
    sd::Tensor<float> output({target_width, target_height, 3, frames});
    for (int64_t f = 0; f < frames; ++f) {
        for (int y = 0; y < target_height; ++y) {
            for (int x = 0; x < target_width; ++x) {
                for (int64_t c = 0; c < 3; ++c) {
                    output.index(x, y, c, f) = (resized.index(x, y, c, f) - mean[c]) / std[c];
                }
            }
        }
    }
    return output;
}

void replace_all_chars(std::string& str, char target, char replacement) {
    std::replace(str.begin(), str.end(), target, replacement);
}

int round_up_to(int value, int base) {
    if (base <= 0) {
        return value;
    }
    return ((value + base - 1) / base) * base;
}

int ed_get_num_physical_cores() {
    const unsigned int n = std::thread::hardware_concurrency();
    return n == 0 ? 4 : static_cast<int>(n);
}

bool file_exists(const std::string& filename) {
    std::error_code ec;
    return std::filesystem::is_regular_file(filename, ec);
}

bool is_directory(const std::string& path) {
    std::error_code ec;
    return std::filesystem::is_directory(path, ec);
}

std::string path_join(const std::string& p1, const std::string& p2) {
    if (p1.empty()) {
        return p2;
    }
    if (p2.empty()) {
        return p1;
    }
    return (std::filesystem::path(p1) / p2).string();
}

std::vector<std::string> split_string(const std::string& str, char delimiter) {
    std::vector<std::string> result;
    std::stringstream ss(str);
    std::string item;
    while (std::getline(ss, item, delimiter)) {
        result.push_back(item);
    }
    return result;
}

void pretty_bytes_progress(int step, int steps, uint64_t bytes_processed, float elapsed_seconds) {
    if (steps <= 0) {
        return;
    }
    const double mb = static_cast<double>(bytes_processed) / (1024.0 * 1024.0);
    std::fprintf(stderr, "\rloading tensors %d/%d %.2fMB %.2fs", step, steps, mb, elapsed_seconds);
    if (step >= steps) {
        std::fprintf(stderr, "\n");
    }
}

void pretty_progress(int step, int steps, float time) {
    if (steps <= 0 || step == 0) {
        return;
    }
    const char* unit = "s/it";
    float speed = time;
    if (speed < 1.0f && speed > 0.0f) {
        speed = 1.0f / speed;
        unit = "it/s";
    }
    std::fprintf(stderr, "\rprogress %d/%d %.2f%s", step, steps, speed, unit);
    if (step >= steps) {
        std::fprintf(stderr, "\n");
    }
}

void log_printf(ed_log_level_t level, const char* file, int line, const char* format, ...) {
    const char* level_name = "debug";
    switch (level) {
        case ED_LOG_INFO: level_name = "info"; break;
        case ED_LOG_WARN: level_name = "warn"; break;
        case ED_LOG_ERROR: level_name = "error"; break;
        case ED_LOG_DEBUG:
        default: break;
    }

    std::fprintf(stderr, "%s:%d [%s] ", file, line, level_name);
    va_list args;
    va_start(args, format);
    std::vfprintf(stderr, format, args);
    va_end(args);
    std::fprintf(stderr, "\n");
}

std::string trim(const std::string& s) {
    const char* ws = " \t\n\r\f\v";
    const size_t begin = s.find_first_not_of(ws);
    if (begin == std::string::npos) {
        return "";
    }
    const size_t end = s.find_last_not_of(ws);
    return s.substr(begin, end - begin + 1);
}

std::unique_ptr<MmapWrapper> MmapWrapper::create(const std::string& filename) {
    static constexpr size_t max_fallback_mmap_size = 256ull * 1024ull * 1024ull;

    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        return nullptr;
    }
    file.seekg(0, std::ios::end);
    const std::streamoff size = file.tellg();
    if (size < 0) {
        return nullptr;
    }
    if (static_cast<uint64_t>(size) > max_fallback_mmap_size) {
        return nullptr;
    }
    file.seekg(0, std::ios::beg);

    std::vector<uint8_t> data(static_cast<size_t>(size));
    if (!data.empty()) {
        file.read(reinterpret_cast<char*>(data.data()), static_cast<std::streamsize>(data.size()));
        if (!file) {
            return nullptr;
        }
    }
    return std::unique_ptr<MmapWrapper>(new MmapWrapper(std::move(data)));
}

bool MmapWrapper::copy_data(void* buf, size_t n, size_t offset) const {
    if (buf == nullptr || offset > data_.size() || n > data_.size() - offset) {
        return false;
    }
    std::memcpy(buf, data_.data() + offset, n);
    return true;
}

bool sd_backend_is(ggml_backend_t backend, const std::string& name) {
    if (backend == nullptr) {
        return false;
    }
    const char* backend_name = ggml_backend_name(backend);
    return backend_name != nullptr && contains(backend_name, name);
}
