#pragma once
// Small dependency-free checksum used only by H3 offline-artifact verification.
#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <string>
#if defined(__linux__)
#include <dlfcn.h>
#include <memory>
#endif
namespace MiniMaxH3 {
class Sha256 {
    std::array<uint32_t,8> h_{{0x6a09e667,0xbb67ae85,0x3c6ef372,0xa54ff53a,0x510e527f,0x9b05688c,0x1f83d9ab,0x5be0cd19}};
    std::array<uint8_t,64> buf_{};
    uint64_t bytes_=0;
    size_t used_=0;
    static uint32_t rr(uint32_t x,unsigned n) {return (x>>n)|(x<<(32-n));}
    void block(const uint8_t* p) {
        static constexpr uint32_t k[]={0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2};
        uint32_t w[64];
        for(int i=0;i<16;++i) w[i]=(uint32_t(p[4*i])<<24)|(uint32_t(p[4*i+1])<<16)|(uint32_t(p[4*i+2])<<8)|p[4*i+3];
        for(int i=16;i<64;++i) w[i]=w[i-16]+(rr(w[i-15],7)^rr(w[i-15],18)^(w[i-15]>>3))+w[i-7]+(rr(w[i-2],17)^rr(w[i-2],19)^(w[i-2]>>10));
        auto s=h_;
        for(int i=0;i<64;++i) {
            uint32_t t1=s[7]+(rr(s[4],6)^rr(s[4],11)^rr(s[4],25))+((s[4]&s[5])^(~s[4]&s[6]))+k[i]+w[i];
            uint32_t t2=(rr(s[0],2)^rr(s[0],13)^rr(s[0],22))+((s[0]&s[1])^(s[0]&s[2])^(s[1]&s[2]));
            s={t1+t2,s[0],s[1],s[2],s[3]+t1,s[4],s[5],s[6]};
        }
        for(int i=0;i<8;++i) h_[i]+=s[i];
    }
public:
    void update(const void* data,size_t size) {
        auto p=static_cast<const uint8_t*>(data);bytes_+=size;
        while(size) {const size_t n=std::min(size,64-used_);std::memcpy(buf_.data()+used_,p,n);used_+=n;p+=n;size-=n;if(used_==64){block(buf_.data());used_=0;}}
    }
    std::string finish() {
        const uint64_t bits=bytes_*8; const uint8_t one=0x80,zero=0;
        update(&one,1); while(used_!=56) update(&zero,1);
        uint8_t tail[8];for(unsigned i=0;i<8;++i)tail[7-i]=uint8_t(bits>>(i*8));update(tail,8);
        std::ostringstream out;out<<std::hex<<std::setfill('0');for(auto v:h_)out<<std::setw(8)<<v;return out.str();
    }
};
// Optional existing-system crypto provider, used only by the opted-in H3/CUDA
// AdaLN identity pass. Full streaming SHA256, no stat/header hash shortcut.
// No mandatory dependency or provider setting changes; portable implementation
// remains the default and the fallback. OpenSSL API: EVP_DigestInit_ex/Update/Final.
inline std::string sha256_file_native_crypto(const std::string& path) {
#if defined(__linux__)
    struct evp_md_ctx_st; struct evp_md_st; struct engine_st;
    using Context = evp_md_ctx_st; using Digest = evp_md_st; using Engine = engine_st;
    struct Library { void* handle; ~Library() { if (handle) dlclose(handle); } } library{dlopen("libcrypto.so.3", RTLD_NOW | RTLD_LOCAL)};
    if (!library.handle) return {};
    auto create = reinterpret_cast<Context* (*)()>(dlsym(library.handle, "EVP_MD_CTX_new"));
    auto destroy = reinterpret_cast<void (*)(Context*)>(dlsym(library.handle, "EVP_MD_CTX_free"));
    auto sha = reinterpret_cast<const Digest* (*)()>(dlsym(library.handle, "EVP_sha256"));
    auto init = reinterpret_cast<int (*)(Context*, const Digest*, Engine*)>(dlsym(library.handle, "EVP_DigestInit_ex"));
    auto update = reinterpret_cast<int (*)(Context*, const void*, size_t)>(dlsym(library.handle, "EVP_DigestUpdate"));
    auto finish = reinterpret_cast<int (*)(Context*, unsigned char*, unsigned int*)>(dlsym(library.handle, "EVP_DigestFinal_ex"));
    if (!create || !destroy || !sha || !init || !update || !finish) return {};
    std::unique_ptr<Context, decltype(destroy)> ctx(create(), destroy);
    if (!ctx || init(ctx.get(), sha(), nullptr) != 1) return {};
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("H3 cannot read artifact: " + path);
    std::array<char, 1024*1024> buffer{};
    while (in) {
        in.read(buffer.data(), buffer.size());
        if (update(ctx.get(), buffer.data(), static_cast<size_t>(in.gcount())) != 1) return {};
    }
    if (!in.eof()) throw std::runtime_error("H3 artifact read failed: " + path);
    std::array<unsigned char, 64> digest{}; unsigned int length = 0;
    if (finish(ctx.get(), digest.data(), &length) != 1 || length != 32) return {};
    std::ostringstream out; out << std::hex << std::setfill('0');
    for (size_t i=0; i<length; ++i) out << std::setw(2) << unsigned(digest[i]);
    return out.str();
#else
    (void)path; return {};
#endif
}
inline std::string sha256_file(const std::string& path, bool use_native_crypto = false) {
    if (use_native_crypto) { auto value = sha256_file_native_crypto(path); if (!value.empty()) return value; }
    std::ifstream in(path,std::ios::binary);if(!in)throw std::runtime_error("H3 cannot read artifact: "+path);
    Sha256 sum;std::array<char,1024*1024> buffer{};
    while(in){in.read(buffer.data(),buffer.size());sum.update(buffer.data(),static_cast<size_t>(in.gcount()));}
    if(!in.eof())throw std::runtime_error("H3 artifact read failed: "+path);
    return sum.finish();
}
} // namespace MiniMaxH3
