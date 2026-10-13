# Include after ggml-cuda and edgedit targets exist. Default OFF, NVIDIA only.
# The pinned ggml worktree is read-only: only a build-local .cu copy is patched.
include_guard(GLOBAL)
option(ED_ENABLE_CUDA_H3_SUBBLOCK "Build H3-only bounded SubBlock CUDA ops" OFF)
if(NOT ED_ENABLE_CUDA_H3_SUBBLOCK)
    return() # no hashing, file generation, target changes or CUDA initialization
endif()
if(NOT ED_GGML_CUDA OR NOT TARGET ggml-cuda OR NOT TARGET edgedit OR ED_GGML_HIP)
    message(FATAL_ERROR "H3 SubBlock requires NVIDIA CUDA and existing ggml-cuda/edgedit targets")
endif()

# This exact source belongs to the project's fixed ggml revision
# f270346d688c7bec7c56bae2ae73343aba6817ad. Hash, not HEAD alone, also detects
# dirty content. Never silently adapt unknown upstream code or mutate its tree.
get_filename_component(_h3_root "${CMAKE_CURRENT_LIST_DIR}/../../../../.." ABSOLUTE)
set(_h3_original "${_h3_root}/third_party/ggml/src/ggml-cuda/ggml-cuda.cu")
set(_h3_expected_sha "1284940278a57c529b88ba4a3b4f49a82cf431fea63848bfb74d9e7f1f6ee265")
if(NOT EXISTS "${_h3_original}")
    message(FATAL_ERROR "H3 SubBlock: missing fixed CUDA dispatcher ${_h3_original}")
endif()
file(SHA256 "${_h3_original}" _h3_actual_sha)
if(NOT _h3_actual_sha STREQUAL _h3_expected_sha)
    message(FATAL_ERROR "H3 SubBlock: fixed CUDA dispatcher SHA256 mismatch; expected ${_h3_expected_sha}, got ${_h3_actual_sha}. Re-audit hooks, do not bypass.")
endif()
# Reconfigure and fail closed if the pinned source changes after configuration.
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${_h3_original}")

function(_h3_replace_once out text anchor replacement)
    string(FIND "${text}" "${anchor}" _first)
    if(_first LESS 0)
        message(FATAL_ERROR "H3 SubBlock: dispatcher anchor missing")
    endif()
    string(LENGTH "${anchor}" _length)
    math(EXPR _after "${_first} + ${_length}")
    string(SUBSTRING "${text}" ${_after} -1 _tail)
    string(FIND "${_tail}" "${anchor}" _second)
    if(NOT _second EQUAL -1)
        message(FATAL_ERROR "H3 SubBlock: dispatcher anchor is not unique")
    endif()
    string(REPLACE "${anchor}" "${replacement}" _result "${text}")
    set(${out} "${_result}" PARENT_SCOPE)
endfunction()

file(READ "${_h3_original}" _h3_text)
set(_h3_anchor "#include \"ggml-cuda/fattn.cuh\"")
_h3_replace_once(_h3_text "${_h3_text}" "${_h3_anchor}"
    "${_h3_anchor}\n#ifdef ED_ENABLE_CUDA_H3_SUBBLOCK\n#include \"h3_subblock.h\"\n#endif")
set(_h3_anchor "#ifdef ED_ENABLE_CUDA_MODULATION\n            if (ed_cuda_fused_modulate_custom_compute(dst, (ed_cuda_modulation_stream_t) ctx.stream())) {\n                break;\n            }\n#endif")
_h3_replace_once(_h3_text "${_h3_text}" "${_h3_anchor}"
    "${_h3_anchor}\n#ifdef ED_ENABLE_CUDA_H3_SUBBLOCK\n            if (ed_cuda_h3_subblock_compute(ctx, dst)) {\n                break;\n            }\n#endif")
set(_h3_anchor "#ifdef ED_ENABLE_CUDA_MODULATION\n            if (ed_cuda_fused_modulate_custom_supported(op)) {\n                return true;\n            }\n#endif")
_h3_replace_once(_h3_text "${_h3_text}" "${_h3_anchor}"
    "${_h3_anchor}\n#ifdef ED_ENABLE_CUDA_H3_SUBBLOCK\n            if (ed_cuda_h3_subblock_supported(dev_ctx->device, op)) {\n                return true;\n            }\n#endif")

# Replace exactly the original compilation unit, not other CUDA files and not
# a second feature's generated dispatcher. Such a conflict needs an explicit
# merge/review instead of one helper silently discarding the other's hooks.
get_target_property(_h3_sources ggml-cuda SOURCES)
get_target_property(_h3_source_dir ggml-cuda SOURCE_DIR)
set(_h3_replacements 0)
set(_h3_new_sources)
set(_h3_generated "${CMAKE_CURRENT_BINARY_DIR}/h3-subblock-generated/ggml-cuda.cu")
foreach(_h3_source IN LISTS _h3_sources)
    get_filename_component(_h3_abs "${_h3_source}" ABSOLUTE BASE_DIR "${_h3_source_dir}")
    if(_h3_abs STREQUAL _h3_original)
        math(EXPR _h3_replacements "${_h3_replacements}+1")
        list(APPEND _h3_new_sources "${_h3_generated}")
    else()
        list(APPEND _h3_new_sources "${_h3_source}")
    endif()
endforeach()
if(NOT _h3_replacements EQUAL 1)
    message(FATAL_ERROR "H3 SubBlock: expected one original CUDA dispatcher in target, found ${_h3_replacements}; possible generated-source conflict")
endif()
# Materialize at configure time: the ggml target belongs to a child directory
# with older GENERATED-property policy, so file(GENERATE) is too late there.
# Only rewrite this build-owned derived file when the content changes.
set(_h3_write TRUE)
if(EXISTS "${_h3_generated}")
    file(READ "${_h3_generated}" _h3_existing)
    if(_h3_existing STREQUAL _h3_text)
        set(_h3_write FALSE)
    endif()
endif()
if(_h3_write)
    file(WRITE "${_h3_generated}" "${_h3_text}")
endif()
set_property(TARGET ggml-cuda PROPERTY SOURCES "${_h3_new_sources}")
set_source_files_properties("${_h3_generated}" PROPERTIES GENERATED TRUE LANGUAGE CUDA)

# CMake in the parent already enables CUDA when building ggml-cuda.
target_sources(ggml-cuda PRIVATE
    "${CMAKE_CURRENT_LIST_DIR}/h3_subblock.cu"
    "${CMAKE_CURRENT_LIST_DIR}/h3_subblock_shape.cpp")
target_compile_definitions(ggml-cuda PRIVATE ED_ENABLE_CUDA_H3_SUBBLOCK)
target_include_directories(ggml-cuda PRIVATE
    "${CMAKE_CURRENT_LIST_DIR}"
    "${_h3_root}/third_party/ggml/src"
    "${_h3_root}/third_party/ggml/src/ggml-cuda")
# Scope the graph adapter to production. Existing CPU-only sparse reference
# tests deliberately retain the expanded-graph oracle without this definition.
target_compile_definitions(edgedit PRIVATE ED_ENABLE_CUDA_H3_SUBBLOCK)
string(SHA256 _h3_generated_sha "${_h3_text}")
file(GENERATE OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/h3-subblock-generated/provenance.txt"
    CONTENT "fixed_ggml_revision=f270346d688c7bec7c56bae2ae73343aba6817ad\noriginal=${_h3_original}\noriginal_sha256=${_h3_actual_sha}\ngenerated=${_h3_generated}\ngenerated_sha256=${_h3_generated_sha}\nhooks=include,custom_compute,custom_supports\nsource_tree_modified=false\n")
message(STATUS "H3 SubBlock: hash-verified build-local CUDA dispatcher; fixed ggml tree unchanged")
