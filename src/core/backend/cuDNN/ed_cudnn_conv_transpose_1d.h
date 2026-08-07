#pragma once

#include "ggml.h"

using ed_cudnn_conv_transpose_1d_stream_t = void *;

enum ed_cudnn_conv_transpose_1d_result_t {
    ED_CUDNN_CONV_TRANSPOSE_1D_SUCCESS = 0,
    ED_CUDNN_CONV_TRANSPOSE_1D_UNSUPPORTED,
    ED_CUDNN_CONV_TRANSPOSE_1D_BUILD_FAILED,
    ED_CUDNN_CONV_TRANSPOSE_1D_EXECUTE_FAILED,
};

ed_cudnn_conv_transpose_1d_result_t ed_cudnn_conv_transpose_1d_compute(
    ggml_tensor * dst, ed_cudnn_conv_transpose_1d_stream_t stream);
