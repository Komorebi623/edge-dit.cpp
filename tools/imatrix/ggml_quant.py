"""ctypes wrapper around the libggml built by edge, directly calling the real q4_K quantize/dequantize path.
Used to offline-verify the effect of imatrix on q4_K quantization error. Does not modify any C++.
"""
import ctypes
import os
import numpy as np

# ggml_type enum (third_party/ggml/include/ggml.h)
GGML_TYPE_F32 = 0
GGML_TYPE_Q8_0 = 8
GGML_TYPE_Q4_K = 12
QK_K = 256

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DEFAULT_LIBDIR = os.environ.get("EDGE_DIT_LIBDIR", os.path.join(_REPO_ROOT, "build-cuda", "bin"))


class GGMLQuant:
    def __init__(self, libdir=DEFAULT_LIBDIR):
        # Load by dependency order. base contains ggml_quantize_chunk / dequantize_row_q4_K
        self.base = ctypes.CDLL(
            os.path.join(libdir, "libggml-base.so"), mode=ctypes.RTLD_GLOBAL
        )

        self.base.ggml_quantize_chunk.restype = ctypes.c_size_t
        self.base.ggml_quantize_chunk.argtypes = [
            ctypes.c_int,                    # type
            ctypes.POINTER(ctypes.c_float),  # src (f32)
            ctypes.c_void_p,                 # dst
            ctypes.c_int64,                  # start
            ctypes.c_int64,                  # nrows
            ctypes.c_int64,                  # n_per_row
            ctypes.POINTER(ctypes.c_float),  # imatrix (length n_per_row, may be NULL)
        ]

        self.base.ggml_row_size.restype = ctypes.c_size_t
        self.base.ggml_row_size.argtypes = [ctypes.c_int, ctypes.c_int64]

        # dequantize_row_q4_K(const block_q4_K* x, float* y, int64_t k)
        self.base.dequantize_row_q4_K.restype = None
        self.base.dequantize_row_q4_K.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_int64,
        ]

    def quantize_q4_K(self, W, imatrix=None):
        """W: [nrows, n_per_row] float32. imatrix: per-input-channel importance of length n_per_row;
        None means take the ref path (equivalent to edge's all-1.0? see notes). Returns the dst buffer (bytes), nrows, n_per_row.

        Note: edge passes an all-1.0 vector (not NULL), taking quantize_row_q4_K_impl;
        NULL takes quantize_row_q4_K_ref (which uses sigma weighting internally, differing from the all-1.0 impl).
        To faithfully reproduce edge's "all-1.0 baseline", this function requires an explicit imatrix by default.
        """
        assert W.ndim == 2
        nrows, n_per_row = W.shape
        assert n_per_row % QK_K == 0, f"n_per_row={n_per_row} not a multiple of {QK_K}"
        Wc = np.ascontiguousarray(W, dtype=np.float32)
        row_size = self.base.ggml_row_size(GGML_TYPE_Q4_K, n_per_row)
        dst = ctypes.create_string_buffer(int(row_size) * int(nrows))

        if imatrix is None:
            im_ptr = ctypes.POINTER(ctypes.c_float)()  # NULL
        else:
            im = np.ascontiguousarray(imatrix, dtype=np.float32)
            assert im.size == n_per_row, f"imatrix length {im.size} != n_per_row {n_per_row}"
            im_ptr = im.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

        ret = self.base.ggml_quantize_chunk(
            GGML_TYPE_Q4_K,
            Wc.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            dst,
            0,
            int(nrows),
            int(n_per_row),
            im_ptr,
        )
        assert ret == int(row_size) * int(nrows), f"quantize returned unexpected {ret}"
        return dst, nrows, n_per_row

    def dequantize_q4_K(self, dst, nrows, n_per_row):
        n = int(nrows) * int(n_per_row)
        out = np.empty(n, dtype=np.float32)
        self.base.dequantize_row_q4_K(
            ctypes.cast(dst, ctypes.c_void_p),
            out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            n,
        )
        return out.reshape(nrows, n_per_row)

    def roundtrip_q4_K(self, W, imatrix=None):
        """Quantize then dequantize, returning the q4_K-reconstructed f32 weights (same shape as W)."""
        dst, nrows, n_per_row = self.quantize_q4_K(W, imatrix)
        return self.dequantize_q4_K(dst, nrows, n_per_row)
