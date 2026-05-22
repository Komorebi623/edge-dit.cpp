#ifndef LIGHT_DIT_H
#define LIGHT_DIT_H

#if defined(_WIN32) || defined(__CYGWIN__)
#  ifdef LD_BUILD_SHARED_LIB
#    ifdef LD_BUILD_DLL
#      define LD_API __declspec(dllexport)
#    else
#      define LD_API __declspec(dllimport)
#    endif
#  else
#    define LD_API
#  endif
#else
#  if defined(__GNUC__) && __GNUC__ >= 4
#    define LD_API __attribute__((visibility("default")))
#  else
#    define LD_API
#  endif
#endif

#ifdef __cplusplus
extern "C" {
#endif

#include <stdbool.h>
#include <stdint.h>

typedef struct ld_context ld_context_t;

typedef enum ld_status_t {
    LD_STATUS_OK = 0,
    LD_STATUS_ERROR,
    LD_STATUS_INVALID_ARGUMENT,
    LD_STATUS_MODEL_LOAD_FAILED,
    LD_STATUS_GENERATION_FAILED,
    LD_STATUS_OUT_OF_MEMORY,
    LD_STATUS_UNSUPPORTED
} ld_status_t;

typedef enum ld_dtype_t {
    LD_DTYPE_AUTO = -1,
    LD_DTYPE_F32  = 0,
    LD_DTYPE_F16  = 1,
    LD_DTYPE_BF16 = 30,
    LD_DTYPE_Q4_0 = 2,
    LD_DTYPE_Q4_1 = 3,
    LD_DTYPE_Q5_0 = 6,
    LD_DTYPE_Q5_1 = 7,
    LD_DTYPE_Q8_0 = 8,
    LD_DTYPE_Q2_K = 10,
    LD_DTYPE_Q3_K = 11,
    LD_DTYPE_Q4_K = 12,
    LD_DTYPE_Q5_K = 13,
    LD_DTYPE_Q6_K = 14
} ld_dtype_t;

typedef enum ld_sampler_t {
    LD_SAMPLER_AUTO = -1,
    LD_SAMPLER_EULER = 0,
    LD_SAMPLER_EULER_A,
    LD_SAMPLER_HEUN,
    LD_SAMPLER_DPM2,
    LD_SAMPLER_DPM_PLUS_PLUS_2S_A,
    LD_SAMPLER_DPM_PLUS_PLUS_2M,
    LD_SAMPLER_DPM_PLUS_PLUS_2M_V2,
    LD_SAMPLER_IPNDM,
    LD_SAMPLER_IPNDM_V,
    LD_SAMPLER_LCM,
    LD_SAMPLER_DDIM_TRAILING,
    LD_SAMPLER_TCD,
    LD_SAMPLER_RES_MULTISTEP,
    LD_SAMPLER_RES_2S,
    LD_SAMPLER_ER_SDE
} ld_sampler_t;

typedef enum ld_scheduler_t {
    LD_SCHEDULER_AUTO = -1,
    LD_SCHEDULER_DISCRETE = 0,
    LD_SCHEDULER_KARRAS,
    LD_SCHEDULER_EXPONENTIAL,
    LD_SCHEDULER_AYS,
    LD_SCHEDULER_GITS,
    LD_SCHEDULER_SGM_UNIFORM,
    LD_SCHEDULER_SIMPLE,
    LD_SCHEDULER_SMOOTHSTEP,
    LD_SCHEDULER_KL_OPTIMAL,
    LD_SCHEDULER_LCM,
    LD_SCHEDULER_BONG_TANGENT
} ld_scheduler_t;

typedef struct ld_image_t {
    uint32_t width;
    uint32_t height;
    uint32_t channels;
    uint8_t * data;
} ld_image_t;

typedef struct ld_image_batch_t {
    ld_image_t * images;
    int count;
} ld_image_batch_t;

typedef struct ld_video_t {
    ld_image_t * frames;
    int frame_count;
} ld_video_t;

typedef struct ld_lora_t {
    const char * path;
    float scale;
    bool high_noise;
} ld_lora_t;

typedef struct ld_context_params_t {
    const char * model_path;

    const char * diffusion_model_path;
    const char * high_noise_diffusion_model_path;
    const char * clip_l_path;
    const char * clip_g_path;
    const char * clip_vision_path;
    const char * t5xxl_path;
    const char * llm_path;
    const char * llm_vision_path;
    const char * vae_path;
    const char * taesd_path;
    const char * control_net_path;

    int n_threads;
    ld_dtype_t weight_type;

    bool use_mmap;
    bool offload_params_to_cpu;
    bool keep_text_encoder_on_cpu;
    bool keep_control_net_on_cpu;
    bool keep_vae_on_cpu;

    bool flash_attention;
    bool diffusion_flash_attention;

    float max_vram_gb;
} ld_context_params_t;

typedef struct ld_sample_params_t {
    ld_sampler_t sampler;
    ld_scheduler_t scheduler;

    int steps;
    float cfg_scale;
    float image_cfg_scale;
    float distilled_guidance;
    float eta;
    float flow_shift;
} ld_sample_params_t;

typedef struct ld_image_generation_params_t {
    const char * prompt;
    const char * negative_prompt;

    int width;
    int height;
    int64_t seed;
    int batch_count;

    const ld_image_t * init_image;
    const ld_image_t * mask_image;
    const ld_image_t * control_image;

    const ld_image_t * ref_images;
    int ref_image_count;

    float strength;
    float control_strength;

    ld_sample_params_t sample;

    const ld_lora_t * loras;
    uint32_t lora_count;
} ld_image_generation_params_t;

typedef struct ld_video_generation_params_t {
    const char * prompt;
    const char * negative_prompt;

    int width;
    int height;
    int frames;
    int64_t seed;

    const ld_image_t * init_image;
    const ld_image_t * end_image;

    const ld_image_t * control_frames;
    int control_frame_count;

    float strength;
    float vace_strength;
    float moe_boundary;

    ld_sample_params_t sample;
    ld_sample_params_t high_noise_sample;

    const ld_lora_t * loras;
    uint32_t lora_count;
} ld_video_generation_params_t;

LD_API void ld_context_params_init(ld_context_params_t * params);
LD_API void ld_sample_params_init(ld_sample_params_t * params);
LD_API void ld_image_generation_params_init(ld_image_generation_params_t * params);
LD_API void ld_video_generation_params_init(ld_video_generation_params_t * params);

LD_API ld_context_t * ld_create_context(const ld_context_params_t * params);
LD_API void ld_free_context(ld_context_t * ctx);

LD_API ld_status_t ld_generate_image(
    ld_context_t * ctx,
    const ld_image_generation_params_t * params,
    ld_image_batch_t * out
);

LD_API ld_status_t ld_generate_video(
    ld_context_t * ctx,
    const ld_video_generation_params_t * params,
    ld_video_t * out
);

LD_API void ld_free_image(ld_image_t * image);
LD_API void ld_free_image_batch(ld_image_batch_t * batch);
LD_API void ld_free_video(ld_video_t * video);

LD_API const char * ld_get_last_error(const ld_context_t * ctx);

#ifdef __cplusplus
}
#endif

#endif /* LIGHT_DIT_H */