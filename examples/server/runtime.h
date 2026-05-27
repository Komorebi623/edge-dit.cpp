#pragma once

#include <cstdint>
#include <mutex>
#include <string>
#include <vector>

#include "light-dit.h"
#include "json.hpp"

using json = nlohmann::json;

struct LightDitServerParams {
    std::string host = "127.0.0.1";
    int port = 8080;
    bool verbose = false;
};

struct LightDitDefaultGenerationParams {
    int width = 1024;
    int height = 1024;
    int steps = 20;
    int64_t seed = -1;
    float cfg_scale = 1.0f;
    float image_cfg_scale = 1.0f;
    float distilled_guidance = 3.5f;
    float flow_shift = 0.0f;
    ld_sampler_t sampler = LD_SAMPLER_AUTO;
    ld_scheduler_t scheduler = LD_SCHEDULER_AUTO;
    ld_cache_mode_t cache_mode = LD_CACHE_DISABLED;
};

struct LightDitServerRuntime {
    ld_context_t* ctx = nullptr;
    std::mutex* ctx_mutex = nullptr;
    const LightDitServerParams* server = nullptr;
    const ld_context_params_t* context = nullptr;
    const LightDitDefaultGenerationParams* defaults = nullptr;
    std::string display_model_path;
};

struct LightDitImageRequest {
    ld_image_generation_params_t params = {};
    std::string prompt;
    std::string negative_prompt;
    std::string cache_scm_mask;
};

std::string ld_status_to_string(ld_status_t status);
std::string ld_cache_mode_to_string(ld_cache_mode_t mode);
bool ld_cache_mode_from_string(const std::string& text, ld_cache_mode_t* mode);
bool ld_sampler_from_string(const std::string& text, ld_sampler_t* sampler);
bool ld_scheduler_from_string(const std::string& text, ld_scheduler_t* scheduler);

std::string base64_encode(const std::vector<uint8_t>& bytes);
bool image_to_png_bytes(const ld_image_t& image, std::vector<uint8_t>* bytes);

bool build_image_request(const json& body,
                         const LightDitServerRuntime& runtime,
                         LightDitImageRequest* request,
                         std::string* error);

json build_capabilities_response(const LightDitServerRuntime& runtime);
