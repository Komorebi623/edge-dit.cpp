cmake_minimum_required(VERSION 3.16)
project(sd_cpp_e2e LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(stable-diffusion REQUIRED CONFIG)
find_package(CUDAToolkit REQUIRED)
find_library(NCCL_LIBRARY nccl REQUIRED)

add_executable(sd_cpp_e2e ../sd_cpp_e2e.cpp)
target_link_libraries(
    sd_cpp_e2e
    PRIVATE
        stable-diffusion
        CUDA::cudart
        CUDA::cublas
        CUDA::culibos
        CUDA::cuda_driver
        ${NCCL_LIBRARY})
