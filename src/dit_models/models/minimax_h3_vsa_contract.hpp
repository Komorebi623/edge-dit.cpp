#pragma once
// SPDX-License-Identifier: Apache-2.0
// Exact payload and quota for the published full H3 FastVideo V1 datafree
// adapter. This is not a generic test for an arbitrary VSA-trained checkpoint.
#include <set>
#include <stdexcept>
#include <string>
namespace MiniMaxH3 { namespace VSAContract {
inline size_t video_topk(size_t tiles) {
    if(!tiles) throw std::invalid_argument("VSA requires video tiles");
    // ceil((1 - 0.9) * tiles), without binary-float off-by-one at multiples of10.
    return tiles/10 + size_t(tiles%10!=0);
}
inline std::set<std::string> ordinary_keys() {
    const std::string prefix="model.diffusion_model.";
    std::set<std::string> result;
    auto pair=[&](const std::string& stem) {
        result.insert(prefix+stem+".lora_A.weight");result.insert(prefix+stem+".lora_B.weight");
    };
    for(int i=0;i<50;++i) {
        const std::string block="blocks."+std::to_string(i)+".";
        for(const auto& suffix:{"attn.qkv_proj","attn.out_proj","mlp.fc1","mlp.fc2","adaln_proj.linear"})pair(block+suffix);
        result.insert(prefix+block+"adaln_proj.linear.bias.delta");
    }
    for(int i=0;i<2;++i)for(const auto& suffix:{"attn.qkv_proj","attn.out_proj","mlp.fc1","mlp.fc2"})
        pair("token_refiner.blocks."+std::to_string(i)+"."+suffix);
    for(const auto& stem:{"video_patch_proj","audio_patch_proj","condition_proj","time_embedder.proj_in","time_embedder.proj_out",
                         "final_layer.adaln_proj.linear","final_layer.video_out","final_layer.audio_out"})
        for(const auto& suffix:{".weight.delta",".bias.delta"})result.insert(prefix+stem+suffix);
    result.insert(prefix+"final_layer.norm.weight.delta");
    for(int i:{3,4,5,6,10,17,19})result.insert(prefix+"blocks."+std::to_string(i)+".norm1.weight.delta");
    for(int i:{4,6,7,8,12,13,15,17})result.insert(prefix+"blocks."+std::to_string(i)+".norm2.weight.delta");
    return result;
}
template<class StorageMap>
inline void validate_ordinary(const StorageMap& tensors) {
    const auto expected=ordinary_keys();
    if(tensors.size()!=expected.size())throw std::invalid_argument("FastH3 VSA V1 incomplete/extra ordinary adapter payload");
    for(const auto& key:expected)if(!tensors.count(key))
        throw std::invalid_argument("FastH3 VSA V1 missing trained update: "+key);
}
}}
