#include "dit_models/models/minimax_h3_sha256.hpp"
#include <cassert>
#include <filesystem>
#include <chrono>
#include <iostream>
int main(){
    MiniMaxH3::Sha256 a;assert(a.finish()=="e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855");
    MiniMaxH3::Sha256 b;b.update("abc",3);assert(b.finish()=="ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
    MiniMaxH3::Sha256 c;for(int i=0;i<1000000;++i)c.update("a",1);assert(c.finish()=="cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0");
    auto folder = std::filesystem::temp_directory_path() / ("h3-sha-native-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    assert(std::filesystem::create_directory(folder));
    for (size_t size : {size_t(0),size_t(1),size_t(55),size_t(56),size_t(63),size_t(64),size_t(65),size_t(1048579)}) {
        std::string bytes(size, '\0'); for (size_t i=0;i<size;++i) bytes[i]=char((i*31+7)%256);
        auto path = folder / std::to_string(size);
        { std::ofstream out(path,std::ios::binary);out.write(bytes.data(),bytes.size()); }
        const auto portable=MiniMaxH3::sha256_file(path.string());
        const auto fast=MiniMaxH3::sha256_file(path.string(),true);
        assert(portable==fast);
        const auto native=MiniMaxH3::sha256_file_native_crypto(path.string());
        assert(native.empty() || native==portable);
        std::filesystem::remove(path); // only this test's newly created file
    }
    bool failed=false;try { MiniMaxH3::sha256_file((folder/"missing").string(),true); }
    catch(const std::runtime_error&) { failed=true; }
    assert(failed);std::filesystem::remove(folder);
    std::cout<<"SHA256 portable/native-provider parity passed (all full file bytes)\n";
}
