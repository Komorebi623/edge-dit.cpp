#ifndef __ED_PARALLEL_CPU_FILE_STORE_HPP__
#define __ED_PARALLEL_CPU_FILE_STORE_HPP__

#include <cstdint>
#include <string>
#include <vector>

namespace edgedit::parallel {

class FileStore {
public:
    FileStore(std::string root, int rank, int world_size);

    const std::string& root() const;

    void write_bytes(const std::string& key, const void* data, size_t size) const;
    std::vector<uint8_t> read_bytes(const std::string& key, size_t expected_size) const;

    void barrier(const std::string& name) const;

private:
    std::string key_path(const std::string& key) const;

    std::string root_;
    int rank_       = 0;
    int world_size_ = 1;
};

} // namespace edgedit::parallel

#endif // __ED_PARALLEL_CPU_FILE_STORE_HPP__
