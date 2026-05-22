cmake_minimum_required(VERSION 3.20)

project(light_dit LANGUAGES C CXX)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(LDIT_BUILD_EXAMPLES "Build examples" ON)

option(LDIT_GGML_CUDA   "Enable ggml CUDA backend" OFF)
option(LDIT_GGML_METAL  "Enable ggml Metal backend" OFF)
option(LDIT_GGML_VULKAN "Enable ggml Vulkan backend" OFF)
option(LDIT_GGML_OPENCL "Enable ggml OpenCL backend" OFF)
option(LDIT_GGML_SYCL   "Enable ggml SYCL backend" OFF)
option(LDIT_GGML_HIP    "Enable ggml HIP backend" OFF)

# Keep ggml clean: do not build ggml's own tests/examples.
set(GGML_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GGML_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)

if(LDIT_GGML_CUDA)
    set(GGML_CUDA ON CACHE BOOL "" FORCE)
endif()

if(LDIT_GGML_METAL)
    set(GGML_METAL ON CACHE BOOL "" FORCE)
endif()

if(LDIT_GGML_VULKAN)
    set(GGML_VULKAN ON CACHE BOOL "" FORCE)
endif()

if(LDIT_GGML_OPENCL)
    set(GGML_OPENCL ON CACHE BOOL "" FORCE)
endif()

if(LDIT_GGML_SYCL)
    set(GGML_SYCL ON CACHE BOOL "" FORCE)
endif()

if(LDIT_GGML_HIP)
    set(GGML_HIP ON CACHE BOOL "" FORCE)
endif()

add_subdirectory(third_party/ggml)

add_library(lightdit STATIC
    src/runtime/ggml/ggml_smoke.cpp
)

target_include_directories(lightdit
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_link_libraries(lightdit
    PUBLIC
        ggml
)

target_compile_features(lightdit PUBLIC cxx_std_17)

if(LDIT_BUILD_EXAMPLES)
    add_executable(smoke_ggml examples/smoke_ggml.cpp)
    target_link_libraries(smoke_ggml PRIVATE lightdit)
endif()