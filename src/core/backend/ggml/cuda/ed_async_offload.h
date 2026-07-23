#pragma once

// Async weight-offload helper: a dedicated CUDA copy stream + events so
// per-segment H2D weight prefetch can overlap with compute on the main stream.
// The public ggml async API is locked to the compute stream, so true overlap
// needs a separate stream reached via raw CUDA here. Header stays CUDA-free
// (opaque void* handles) so it can be included from the pure C++ header
// ggml_extend.hpp.

#include "ggml.h"
#include "ggml-backend.h"

using ed_copy_stream_t = void *;  // cudaStream_t
using ed_copy_event_t  = void *;  // cudaEvent_t

// Create/destroy a non-blocking copy stream on the given CUDA device.
ed_copy_stream_t ed_async_offload_stream_create(int device);
void             ed_async_offload_stream_destroy(ed_copy_stream_t stream);

// Create/destroy a CUDA event (used to signal H2D completion).
ed_copy_event_t ed_async_offload_event_create();
void            ed_async_offload_event_destroy(ed_copy_event_t event);

// Async host->device copy issued on the copy stream (does not synchronize).
void ed_async_offload_h2d(void * dst, const void * src, size_t nbytes, ed_copy_stream_t stream);

// Record the event on the copy stream (after all H2D for a slot are issued).
void ed_async_offload_event_record(ed_copy_event_t event, ed_copy_stream_t stream);

// Block the host until the event completes (Ladder ii: no compute overlap).
void ed_async_offload_event_synchronize(ed_copy_event_t event);

// Make compute_backend's stream wait on the event (Ladder iii: true overlap).
void ed_async_offload_compute_wait(ggml_backend_t compute_backend, ed_copy_event_t event);
