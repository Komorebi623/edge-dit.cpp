cmake_minimum_required(VERSION 3.20)

project(edge_dit LANGUAGES C CXX)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT XCODE AND NOT MSVC AND NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS Debug Release MinSizeRel RelWithDebInfo)
endif()

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_compile_definitions(GGML_MAX_NAME=128)

find_package(Threads REQUIRED)

option(ED_BUILD_EXAMPLES "Build examples" ON)
option(ED_BUILD_TESTS "Build edge-dit tests" OFF)
option(ED_BUILD_SHARED_LIBS "Build edge-dit as a shared library" OFF)

if(ED_BUILD_TESTS)
    enable_testing()
endif()

option(ED_GGML_CUDA   "Enable ggml CUDA backend" OFF)
option(ED_GGML_METAL  "Enable ggml Metal backend" OFF)
option(ED_GGML_VULKAN "Enable ggml Vulkan backend" OFF)
option(ED_GGML_OPENCL "Enable ggml OpenCL backend" OFF)
option(ED_GGML_SYCL   "Enable ggml SYCL backend" OFF)
option(ED_GGML_HIP    "Enable ggml HIP backend" OFF)

if(MSVC)
    add_compile_definitions(_CRT_SECURE_NO_WARNINGS)
    add_compile_options(
        $<$<COMPILE_LANGUAGE:C>:/MP>
        $<$<COMPILE_LANGUAGE:CXX>:/MP>
    )
endif()

# Keep ggml clean: do not build ggml's own tests/examples.
set(GGML_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GGML_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)

if(ED_GGML_CUDA)
    set(GGML_CUDA ON CACHE BOOL "" FORCE)
endif()

if(ED_GGML_METAL)
    set(GGML_METAL ON CACHE BOOL "" FORCE)
endif()

if(ED_GGML_VULKAN)
    set(GGML_VULKAN ON CACHE BOOL "" FORCE)
endif()

if(ED_GGML_OPENCL)
    set(GGML_OPENCL ON CACHE BOOL "" FORCE)
endif()

if(ED_GGML_SYCL)
    set(GGML_SYCL ON CACHE BOOL "" FORCE)
endif()

if(ED_GGML_HIP)
    set(GGML_HIP ON CACHE BOOL "" FORCE)
endif()

add_subdirectory(third_party/ggml)

file(GLOB_RECURSE ED_CORE_SOURCES CONFIGURE_DEPENDS
    "src/core/*.cpp"
    "src/core/*.h"
    "src/core/*.hpp"
)

file(GLOB ED_API_SOURCES CONFIGURE_DEPENDS
    "src/*.cpp"
    "src/*.h"
    "src/*.hpp"
)

file(GLOB ED_UTIL_SOURCES CONFIGURE_DEPENDS
    "src/utils/*.cpp"
    "src/utils/*.h"
    "src/utils/*.hpp"
)

file(GLOB_RECURSE ED_DIT_MODEL_SOURCES CONFIGURE_DEPENDS
    "src/dit_models/*.cpp"
    "src/dit_models/*.h"
    "src/dit_models/*.hpp"
)
list(FILTER ED_DIT_MODEL_SOURCES EXCLUDE REGEX ".*/super_resolution/.*")

set(ED_SOURCES
    ${ED_API_SOURCES}
    ${ED_CORE_SOURCES}
    ${ED_UTIL_SOURCES}
    ${ED_DIT_MODEL_SOURCES}
    "src/utils/model_io/binary_io.h"
    "src/utils/model_io/gguf_io.cpp"
    "src/utils/model_io/gguf_io.h"
    "src/utils/model_io/gguf_reader_ext.h"
    "src/utils/model_io/safetensors_io.cpp"
    "src/utils/model_io/safetensors_io.h"
    "src/utils/model_io/tensor_storage.h"
)

if(ED_BUILD_SHARED_LIBS)
    add_library(edgedit SHARED ${ED_SOURCES})
    target_compile_definitions(edgedit PRIVATE ED_BUILD_DLL)
    target_compile_definitions(edgedit PUBLIC ED_BUILD_SHARED_LIB)
else()
    add_library(edgedit STATIC ${ED_SOURCES})
endif()

target_include_directories(edgedit
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${CMAKE_CURRENT_SOURCE_DIR}/src/core
        ${CMAKE_CURRENT_SOURCE_DIR}/third_party
        ${CMAKE_CURRENT_SOURCE_DIR}/third_party/ggml/src
)

target_link_libraries(edgedit
    PUBLIC
        ggml
)

target_compile_features(edgedit PUBLIC cxx_std_17)

if(ED_BUILD_EXAMPLES)
    add_executable(ed-cli examples/cli/main.cpp)
    target_link_libraries(ed-cli PRIVATE edgedit)
    target_include_directories(ed-cli PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/third_party)
    target_compile_features(ed-cli PRIVATE cxx_std_17)

    add_executable(ed-server
        examples/server/main.cpp
        examples/server/routes.cpp
        examples/server/runtime.cpp
    )
    target_link_libraries(ed-server PRIVATE edgedit Threads::Threads)
    target_include_directories(ed-server PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/third_party)
    target_compile_features(ed-server PRIVATE cxx_std_17)
endif()

if(ED_BUILD_TESTS)
    add_executable(cache-smoke
        tests/cache_smoke.cpp
        src/core/optimization/cache/cache_config.cpp
        src/core/optimization/cache/cache_methods.cpp
        src/core/optimization/cache/cache_runtime.cpp
        src/core/optimization/cache/cache_types.cpp
    )
    target_include_directories(cache-smoke
        PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/include
            ${CMAKE_CURRENT_SOURCE_DIR}/src
            ${CMAKE_CURRENT_SOURCE_DIR}/src/core
            ${CMAKE_CURRENT_SOURCE_DIR}/third_party
            ${CMAKE_CURRENT_SOURCE_DIR}/third_party/ggml/include
    )
    target_compile_features(cache-smoke PRIVATE cxx_std_17)
    add_test(NAME cache-smoke COMMAND cache-smoke)
endif()

set(ED_PUBLIC_HEADERS include/edge-dit.h)
set_target_properties(edgedit PROPERTIES PUBLIC_HEADER "${ED_PUBLIC_HEADERS}")

install(TARGETS edgedit
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION bin
    PUBLIC_HEADER DESTINATION include
)

if(ED_BUILD_EXAMPLES)
    install(TARGETS ed-cli ed-server RUNTIME DESTINATION bin)
endif()
