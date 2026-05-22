#ifndef LD_MODEL_LOADER_H
#define LD_MODEL_LOADER_H

#include <functional>
#include <map>
#include <set>
#include <string>
#include <vector>

#include "ggml-backend.h"
#include "ggml.h"
#include "utils/model_io/tensor_storage.h"
#include "utils/ordered_map.hpp"

enum SDVersion {
    VERSION_SD1,
    VERSION_SD1_INPAINT,
    VERSION_SD1_PIX2PIX,
    VERSION_SD1_TINY_UNET,
    VERSION_SD2,
    VERSION_SD2_INPAINT,
    VERSION_SD2_TINY_UNET,
    VERSION_SDXS_512_DS,
    VERSION_SDXS_09,
    VERSION_SDXL,
    VERSION_SDXL_INPAINT,
    VERSION_SDXL_PIX2PIX,
    VERSION_SDXL_VEGA,
    VERSION_SDXL_SSD1B,
    VERSION_SVD,
    VERSION_SD3,
    VERSION_FLUX,
    VERSION_FLUX_FILL,
    VERSION_FLUX_CONTROLS,
    VERSION_FLEX_2,
    VERSION_CHROMA_RADIANCE,
    VERSION_WAN2,
    VERSION_WAN2_2_I2V,
    VERSION_WAN2_2_TI2V,
    VERSION_QWEN_IMAGE,
    VERSION_ANIMA,
    VERSION_FLUX2,
    VERSION_FLUX2_KLEIN,
    VERSION_Z_IMAGE,
    VERSION_OVIS_IMAGE,
    VERSION_ERNIE_IMAGE,
    VERSION_COUNT,
};

static inline bool ld_version_is_sd1(SDVersion version) {
    return version == VERSION_SD1 || version == VERSION_SD1_INPAINT ||
           version == VERSION_SD1_PIX2PIX || version == VERSION_SD1_TINY_UNET ||
           version == VERSION_SDXS_512_DS;
}

static inline bool ld_version_is_sd2(SDVersion version) {
    return version == VERSION_SD2 || version == VERSION_SD2_INPAINT ||
           version == VERSION_SD2_TINY_UNET || version == VERSION_SDXS_09;
}

static inline bool ld_version_is_sdxl(SDVersion version) {
    return version == VERSION_SDXL || version == VERSION_SDXL_INPAINT ||
           version == VERSION_SDXL_PIX2PIX || version == VERSION_SDXL_SSD1B ||
           version == VERSION_SDXL_VEGA;
}

static inline bool ld_version_is_unet(SDVersion version) {
    return ld_version_is_sd1(version) || ld_version_is_sd2(version) || ld_version_is_sdxl(version);
}

static inline bool ld_version_is_sd3(SDVersion version) {
    return version == VERSION_SD3;
}

static inline bool ld_version_is_flux(SDVersion version) {
    return version == VERSION_FLUX || version == VERSION_FLUX_FILL ||
           version == VERSION_FLUX_CONTROLS || version == VERSION_FLEX_2 ||
           version == VERSION_OVIS_IMAGE || version == VERSION_CHROMA_RADIANCE;
}

static inline bool ld_version_is_flux2(SDVersion version) {
    return version == VERSION_FLUX2 || version == VERSION_FLUX2_KLEIN;
}

static inline bool ld_version_is_wan(SDVersion version) {
    return version == VERSION_WAN2 || version == VERSION_WAN2_2_I2V || version == VERSION_WAN2_2_TI2V;
}

static inline bool ld_version_is_qwen_image(SDVersion version) {
    return version == VERSION_QWEN_IMAGE;
}

static inline bool ld_version_is_anima(SDVersion version) {
    return version == VERSION_ANIMA;
}

static inline bool ld_version_is_z_image(SDVersion version) {
    return version == VERSION_Z_IMAGE;
}

static inline bool ld_version_is_ernie_image(SDVersion version) {
    return version == VERSION_ERNIE_IMAGE;
}

static inline bool ld_version_uses_flux2_vae(SDVersion version) {
    return ld_version_is_flux2(version) || ld_version_is_ernie_image(version);
}

static inline bool ld_version_is_inpaint(SDVersion version) {
    return version == VERSION_SD1_INPAINT || version == VERSION_SD2_INPAINT ||
           version == VERSION_SDXL_INPAINT || version == VERSION_FLUX_FILL ||
           version == VERSION_FLEX_2;
}

static inline bool ld_version_is_dit(SDVersion version) {
    return ld_version_is_flux(version) || ld_version_is_flux2(version) ||
           ld_version_is_sd3(version) || ld_version_is_wan(version) ||
           ld_version_is_qwen_image(version) || ld_version_is_anima(version) ||
           ld_version_is_z_image(version) || ld_version_is_ernie_image(version);
}

static inline bool ld_version_is_unet_edit(SDVersion version) {
    return version == VERSION_SD1_PIX2PIX || version == VERSION_SDXL_PIX2PIX;
}

static inline bool ld_version_is_control(SDVersion version) {
    return version == VERSION_FLUX_CONTROLS || version == VERSION_FLEX_2;
}

using String2TensorStorage = OrderedMap<std::string, TensorStorage>;
using TensorTypeRules = std::vector<std::pair<std::string, ggml_type>>;

TensorTypeRules parse_tensor_type_rules(const std::string& tensor_type_rules);
const char* ld_version_name(SDVersion version);

class ModelLoader {
public:
    bool init_from_file(const std::string& file_path, const std::string& prefix = "");
    void convert_tensors_name();
    bool init_from_file_and_convert_name(const std::string& file_path,
                                         const std::string& prefix = "",
                                         SDVersion version = VERSION_COUNT);

    SDVersion get_ld_version();
    std::map<ggml_type, uint32_t> get_wtype_stat() const;
    std::map<ggml_type, uint32_t> get_conditioner_wtype_stat() const;
    std::map<ggml_type, uint32_t> get_diffusion_model_wtype_stat() const;
    std::map<ggml_type, uint32_t> get_vae_wtype_stat() const;

    String2TensorStorage& get_tensor_storage_map() { return tensor_storage_map_; }
    const String2TensorStorage& get_tensor_storage_map() const { return tensor_storage_map_; }
    const std::vector<std::string>& get_file_paths() const { return file_paths_; }
    const std::string& get_last_error() const { return last_error_; }

    std::vector<std::string> get_tensor_names() const;

    void set_wtype_override(ggml_type wtype, std::string tensor_type_rules = "");
    bool load_tensors(on_new_tensor_cb_t on_new_tensor_cb, int n_threads = 0, bool use_mmap = false);
    bool load_tensors(std::map<std::string, ggml_tensor*>& tensors,
                      std::set<std::string> ignore_tensors = {},
                      int n_threads = 0,
                      bool use_mmap = false);

    bool tensor_should_be_converted(const TensorStorage& tensor_storage, ggml_type type) const;
    int64_t get_params_mem_size(ggml_backend_t backend, ggml_type type = GGML_TYPE_COUNT) const;

private:
    SDVersion version_ = VERSION_COUNT;
    std::vector<std::string> file_paths_;
    String2TensorStorage tensor_storage_map_;
    std::string last_error_;

    void clear();
    void set_error(const std::string& error);
    void add_tensor_storage(const TensorStorage& tensor_storage);

    bool init_from_gguf_file(const std::string& file_path, const std::string& prefix = "");
    bool init_from_safetensors_file(const std::string& file_path, const std::string& prefix = "");
    bool init_from_safetensors_index_file(const std::string& file_path, const std::string& prefix = "");
    bool init_from_diffusers_directory(const std::string& dir_path, const std::string& prefix = "");
};

#endif
