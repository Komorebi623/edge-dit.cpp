#ifndef LD_UTIL_H
#define LD_UTIL_H

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "ggml-backend.h"

#define SAFE_STR(s) ((s) ? (s) : "")
#define BOOL_STR(b) ((b) ? "true" : "false")

enum ld_log_level_t {
    LD_LOG_DEBUG = 0,
    LD_LOG_INFO,
    LD_LOG_WARN,
    LD_LOG_ERROR,
};

bool ends_with(const std::string& str, const std::string& ending);
bool starts_with(const std::string& str, const std::string& start);
bool contains(const std::string& str, const std::string& substr);

std::string sd_format(const char* fmt, ...);

void replace_all_chars(std::string& str, char target, char replacement);

int round_up_to(int value, int base);
int ld_get_num_physical_cores();

bool file_exists(const std::string& filename);
bool is_directory(const std::string& path);

std::string path_join(const std::string& p1, const std::string& p2);
std::vector<std::string> split_string(const std::string& str, char delimiter);
void pretty_bytes_progress(int step, int steps, uint64_t bytes_processed, float elapsed_seconds);

void log_printf(ld_log_level_t level, const char* file, int line, const char* format, ...);

std::string trim(const std::string& s);

class MmapWrapper {
public:
    static std::unique_ptr<MmapWrapper> create(const std::string& filename);

    virtual ~MmapWrapper() = default;

    MmapWrapper(const MmapWrapper&)            = delete;
    MmapWrapper& operator=(const MmapWrapper&) = delete;
    MmapWrapper(MmapWrapper&&)                 = delete;
    MmapWrapper& operator=(MmapWrapper&&)      = delete;

    const uint8_t* data() const { return data_.data(); }
    size_t size() const { return data_.size(); }
    bool copy_data(void* buf, size_t n, size_t offset) const;

private:
    explicit MmapWrapper(std::vector<uint8_t> data)
        : data_(std::move(data)) {}

    std::vector<uint8_t> data_;
};

bool sd_backend_is(ggml_backend_t backend, const std::string& name);

#define LOG_DEBUG(format, ...) log_printf(LD_LOG_DEBUG, __FILE__, __LINE__, format, ##__VA_ARGS__)
#define LOG_INFO(format, ...) log_printf(LD_LOG_INFO, __FILE__, __LINE__, format, ##__VA_ARGS__)
#define LOG_WARN(format, ...) log_printf(LD_LOG_WARN, __FILE__, __LINE__, format, ##__VA_ARGS__)
#define LOG_ERROR(format, ...) log_printf(LD_LOG_ERROR, __FILE__, __LINE__, format, ##__VA_ARGS__)

#endif
