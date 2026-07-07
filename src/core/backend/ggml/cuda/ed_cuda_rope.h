#pragma once

#include "ggml.h"

using ed_cuda_rope_stream_t = void *;

bool ed_cuda_rope_custom_supported(const ggml_tensor * dst);

bool ed_cuda_rope_custom_compute(ggml_tensor * dst, ed_cuda_rope_stream_t stream);
