#include "utils/util.h"

#include <algorithm>
#include <cstdarg>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <thread>

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

void replace_all_chars(std::string& str, char target, char replacement) {
    std::replace(str.begin(), str.end(), target, replacement);
}

int round_up_to(int value, int base) {
    if (base <= 0) {
        return value;
    }
    return ((value + base - 1) / base) * base;
}

int ld_get_num_physical_cores() {
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

void log_printf(ld_log_level_t level, const char* file, int line, const char* format, ...) {
    const char* level_name = "debug";
    switch (level) {
        case LD_LOG_INFO: level_name = "info"; break;
        case LD_LOG_WARN: level_name = "warn"; break;
        case LD_LOG_ERROR: level_name = "error"; break;
        case LD_LOG_DEBUG:
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
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        return nullptr;
    }
    file.seekg(0, std::ios::end);
    const std::streamoff size = file.tellg();
    if (size < 0) {
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
