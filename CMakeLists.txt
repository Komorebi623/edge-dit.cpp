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
option(ED_ENABLE_PARALLEL "Enable edge-dit lightweight parallel runtime" ON)
option(ED_ENABLE_NCCL "Enable NCCL communication backend" OFF)
option(ED_ENABLE_MPI "Enable MPI bootstrap for distributed process groups" OFF)
set(NCCL_ROOT "" CACHE PATH "Path to NCCL installation")

if(ED_BUILD_TESTS)
    enable_testing()
    find_package(Python3 COMPONENTS Interpreter REQUIRED)
endif()

option(ED_GGML_CUDA   "Enable ggml CUDA backend" OFF)
option(ED_GGML_METAL  "Enable ggml Metal backend" OFF)
option(ED_GGML_VULKAN "Enable ggml Vulkan backend" OFF)
option(ED_GGML_OPENCL "Enable ggml OpenCL backend" OFF)
option(ED_GGML_SYCL   "Enable ggml SYCL backend" OFF)
option(ED_GGML_HIP    "Enable ggml HIP backend" OFF)

if(ED_ENABLE_NCCL)
    find_package(CUDAToolkit REQUIRED)

    set(_ED_NCCL_HINTS)
    if(NCCL_ROOT)
        list(APPEND _ED_NCCL_HINTS "${NCCL_ROOT}")
    endif()
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/third_party/nccl")
        list(APPEND _ED_NCCL_HINTS "${CMAKE_CURRENT_SOURCE_DIR}/third_party/nccl")
    endif()

    find_path(ED_NCCL_INCLUDE_DIR
        NAMES nccl.h
        HINTS ${_ED_NCCL_HINTS}
        PATH_SUFFIXES include
    )
    find_library(ED_NCCL_LIBRARY
        NAMES nccl
        HINTS ${_ED_NCCL_HINTS}
        PATH_SUFFIXES lib lib64
    )

    if(NOT ED_NCCL_INCLUDE_DIR OR NOT ED_NCCL_LIBRARY)
        message(FATAL_ERROR "ED_ENABLE_NCCL=ON requires NCCL. Set NCCL_ROOT or provide third_party/nccl.")
    endif()
endif()

if(ED_ENABLE_MPI)
    find_package(MPI REQUIRED COMPONENTS CXX)
endif()

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

if(NOT ED_ENABLE_NCCL)
    list(FILTER ED_SOURCES EXCLUDE REGEX ".*/src/core/parallel/backends/nccl/.*")
endif()

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

if(ED_ENABLE_PARALLEL)
    target_compile_definitions(edgedit PUBLIC ED_ENABLE_PARALLEL)
endif()

if(ED_ENABLE_NCCL)
    target_compile_definitions(edgedit PUBLIC ED_ENABLE_NCCL)
    target_include_directories(edgedit PRIVATE ${ED_NCCL_INCLUDE_DIR})
    target_link_libraries(edgedit PRIVATE CUDA::cudart ${ED_NCCL_LIBRARY})
endif()

if(ED_ENABLE_MPI)
    target_compile_definitions(edgedit PUBLIC ED_ENABLE_MPI)
    target_link_libraries(edgedit PUBLIC MPI::MPI_CXX)
endif()

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
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/cache_smoke.cpp")
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

    add_executable(parallel-collective-test
        tests/test_process_group.cpp
    )
    target_link_libraries(parallel-collective-test PRIVATE edgedit)
    target_include_directories(parallel-collective-test
        PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/src
            ${CMAKE_CURRENT_SOURCE_DIR}/src/core
    )
    target_compile_features(parallel-collective-test PRIVATE cxx_std_17)
    
    add_executable(ggml-comm-test
        tests/ggml_comm_test.cpp
    )

    target_link_libraries(ggml-comm-test PRIVATE edgedit)

    target_include_directories(ggml-comm-test
        PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/include
            ${CMAKE_CURRENT_SOURCE_DIR}/src
            ${CMAKE_CURRENT_SOURCE_DIR}/src/core
            ${CMAKE_CURRENT_SOURCE_DIR}/third_party
            ${CMAKE_CURRENT_SOURCE_DIR}/third_party/ggml/include
    )

    target_compile_features(ggml-comm-test PRIVATE cxx_std_17)

    add_executable(graph-cut-comm-test
        tests/graph_cut_comm_test.cpp
    )

    target_link_libraries(graph-cut-comm-test PRIVATE edgedit)

    target_include_directories(graph-cut-comm-test
        PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/include
            ${CMAKE_CURRENT_SOURCE_DIR}/src
            ${CMAKE_CURRENT_SOURCE_DIR}/src/core
            ${CMAKE_CURRENT_SOURCE_DIR}/third_party
            ${CMAKE_CURRENT_SOURCE_DIR}/third_party/ggml/include
    )

    target_compile_features(graph-cut-comm-test PRIVATE cxx_std_17)

    if(ED_ENABLE_NCCL)
        target_include_directories(parallel-collective-test PRIVATE ${ED_NCCL_INCLUDE_DIR})
        target_link_libraries(parallel-collective-test PRIVATE CUDA::cudart ${ED_NCCL_LIBRARY})
    endif()
    if(ED_ENABLE_MPI)
        target_link_libraries(parallel-collective-test PRIVATE MPI::MPI_CXX)
    endif()
    add_test(
        NAME parallel-collective-cpu
        COMMAND
            ${Python3_EXECUTABLE}
            ${CMAKE_CURRENT_SOURCE_DIR}/scripts/run_parallel_collective_test.py
            --binary $<TARGET_FILE:parallel-collective-test>
            --backend cpu
            --world-size 2
    )
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
