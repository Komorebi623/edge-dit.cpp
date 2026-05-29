#include "parallel/backends/cpu/file_store.hpp"

#include <chrono>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <thread>

namespace edgedit::parallel {
namespace {

constexpr int kPollMs     = 10;
constexpr int kTimeoutSec = 120;

void wait_for_path(const std::filesystem::path& path) {
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(kTimeoutSec);
    while (!std::filesystem::exists(path)) {
        if (std::chrono::steady_clock::now() > deadline) {
            throw std::runtime_error("timed out waiting for file store path: " + path.string());
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(kPollMs));
    }
}

} // namespace

FileStore::FileStore(std::string root, int rank, int world_size)
    : root_(std::move(root)), rank_(rank), world_size_(world_size) {
    if (root_.empty()) {
        throw std::invalid_argument("parallel file store path must not be empty");
    }
    std::filesystem::create_directories(root_);
}

const std::string& FileStore::root() const {
    return root_;
}

void FileStore::write_bytes(const std::string& key, const void* data, size_t size) const {
    const auto final_path = std::filesystem::path(key_path(key));
    const auto tmp_path   = final_path.string() + ".tmp." + std::to_string(rank_);
    {
        std::ofstream out(tmp_path, std::ios::binary | std::ios::trunc);
        if (!out) {
            throw std::runtime_error("failed to open file store tmp path for write: " + tmp_path);
        }
        out.write(reinterpret_cast<const char*>(data), static_cast<std::streamsize>(size));
        if (!out) {
            throw std::runtime_error("failed to write file store payload: " + tmp_path);
        }
    }
    std::filesystem::rename(tmp_path, final_path);
}

std::vector<uint8_t> FileStore::read_bytes(const std::string& key, size_t expected_size) const {
    const auto path = std::filesystem::path(key_path(key));
    wait_for_path(path);

    std::ifstream in(path, std::ios::binary);
    if (!in) {
        throw std::runtime_error("failed to open file store payload: " + path.string());
    }

    std::vector<uint8_t> bytes(expected_size);
    in.read(reinterpret_cast<char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
    if (static_cast<size_t>(in.gcount()) != expected_size) {
        throw std::runtime_error("unexpected file store payload size: " + path.string());
    }
    return bytes;
}

void FileStore::barrier(const std::string& name) const {
    const std::string prefix = "barrier_" + name + "_";
    const uint8_t token     = 1;
    write_bytes(prefix + std::to_string(rank_), &token, sizeof(token));

    for (int r = 0; r < world_size_; ++r) {
        (void) read_bytes(prefix + std::to_string(r), sizeof(token));
    }
}

std::string FileStore::key_path(const std::string& key) const {
    return (std::filesystem::path(root_) / key).string();
}

} // namespace edgedit::parallel
