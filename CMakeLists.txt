cmake_minimum_required(VERSION 3.20)

project(light_dit LANGUAGES C CXX)

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

option(LDIT_BUILD_EXAMPLES "Build examples" ON)
option(LDIT_BUILD_SHARED_LIBS "Build light-dit as a shared library" OFF)

option(LDIT_GGML_CUDA   "Enable ggml CUDA backend" OFF)
option(LDIT_GGML_METAL  "Enable ggml Metal backend" OFF)
option(LDIT_GGML_VULKAN "Enable ggml Vulkan backend" OFF)
option(LDIT_GGML_OPENCL "Enable ggml OpenCL backend" OFF)
option(LDIT_GGML_SYCL   "Enable ggml SYCL backend" OFF)
option(LDIT_GGML_HIP    "Enable ggml HIP backend" OFF)

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

file(GLOB_RECURSE LDIT_CORE_SOURCES CONFIGURE_DEPENDS
    "src/core/*.cpp"
    "src/core/*.h"
    "src/core/*.hpp"
)

file(GLOB LDIT_API_SOURCES CONFIGURE_DEPENDS
    "src/*.cpp"
    "src/*.h"
    "src/*.hpp"
)

file(GLOB LDIT_UTIL_SOURCES CONFIGURE_DEPENDS
    "src/utils/*.cpp"
    "src/utils/*.h"
    "src/utils/*.hpp"
)

file(GLOB_RECURSE LDIT_DIT_MODEL_SOURCES CONFIGURE_DEPENDS
    "src/dit_models/*.cpp"
    "src/dit_models/*.h"
    "src/dit_models/*.hpp"
)
list(FILTER LDIT_DIT_MODEL_SOURCES EXCLUDE REGEX ".*/super_resolution/.*")

set(LDIT_SOURCES
    ${LDIT_API_SOURCES}
    ${LDIT_CORE_SOURCES}
    ${LDIT_UTIL_SOURCES}
    ${LDIT_DIT_MODEL_SOURCES}
    "src/utils/model_io/binary_io.h"
    "src/utils/model_io/gguf_io.cpp"
    "src/utils/model_io/gguf_io.h"
    "src/utils/model_io/gguf_reader_ext.h"
    "src/utils/model_io/safetensors_io.cpp"
    "src/utils/model_io/safetensors_io.h"
    "src/utils/model_io/tensor_storage.h"
)

if(LDIT_BUILD_SHARED_LIBS)
    add_library(lightdit SHARED ${LDIT_SOURCES})
    target_compile_definitions(lightdit PRIVATE LD_BUILD_DLL)
    target_compile_definitions(lightdit PUBLIC LD_BUILD_SHARED_LIB)
else()
    add_library(lightdit STATIC ${LDIT_SOURCES})
endif()

target_include_directories(lightdit
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${CMAKE_CURRENT_SOURCE_DIR}/src/core
        ${CMAKE_CURRENT_SOURCE_DIR}/third_party
        ${CMAKE_CURRENT_SOURCE_DIR}/third_party/ggml/src
)

target_link_libraries(lightdit
    PUBLIC
        ggml
)

target_compile_features(lightdit PUBLIC cxx_std_17)

if(LDIT_BUILD_EXAMPLES)
    add_executable(ld-cli examples/cli/main.cpp)
    target_link_libraries(ld-cli PRIVATE lightdit)
    target_include_directories(ld-cli PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/third_party)
    target_compile_features(ld-cli PRIVATE cxx_std_17)
endif()

set(LDIT_PUBLIC_HEADERS include/light-dit.h)
set_target_properties(lightdit PROPERTIES PUBLIC_HEADER "${LDIT_PUBLIC_HEADERS}")

install(TARGETS lightdit
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION bin
    PUBLIC_HEADER DESTINATION include
)

if(LDIT_BUILD_EXAMPLES)
    install(TARGETS ld-cli RUNTIME DESTINATION bin)
endif()
