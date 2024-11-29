cmake_minimum_required(VERSION 3.20)

# set(CMAKE_CUDA_COMPILER /usr/local/cuda-11.8/bin/nvcc)

project(RTSearch C CXX CUDA)

set(CMAKE_CUDA_ARCHITECTURES 80 89)

set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)
set(LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

find_package(OptiX REQUIRED)
if(OptiX_FOUND)
    include_directories(${OptiX_INCLUDE_DIR})
    message(STATUS "Optix Found")
    message(STATUS ${OptiX_INCLUDE_DIR})
else()
    message(FATAL_ERROR "Optix cannot found")
endif()

find_package(CUDA REQUIRED)
if(CUDA_FOUND)
    include_directories(${CUDA_INCLUDE_DIRS})
    message(STATUS "CUDA Found")
    message(STATUS ${CUDA_INCLUDE_DIRS})
else()
    message(FATAL_ERROR "CUDA cannot found")
endif()

find_package(OpenMP REQUIRED)

include(${PROJECT_SOURCE_DIR}/cmake/configure_optix.cmake)

cuda_compile_and_embed(embedded_ptx_code DevicePrograms.cu)

# add_executable(rt_search
#     ${embedded_ptx_code}
#     main.cpp
#     RTBase.cpp
#     simple_bs.cu
# )

# target_compile_options(rt_search PRIVATE $<$<COMPILE_LANGUAGE:CXX>:
# 	-Wall -Wextra -Wpedantic
# 	${OpenMP_CXX_FLAGS}
# 	-std=c++14
# >)

# target_compile_options(rt_search PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
#     --extended-lambda
#     --use_fast_math
#     --expt-relaxed-constexpr
#     -O3
#     -Xcompiler=-fopenmp
#     -std=c++14
#     -gencode arch=compute_75,code=sm_75
# >)

# target_link_libraries(rt_search
#     ${CUDA_LIBRARIES}
#     ${CUDA_CUDA_LIBRARY}
#     OpenMP::OpenMP_CXX
# )

# ========================
add_executable(measure
    ${embedded_ptx_code}
    measure.cpp
    RTBase.cpp
    simple_bs.cu
    # access.cu
    casting_kernels.cu
)

target_include_directories(measure PRIVATE
    ${CMAKE_SOURCE_DIR}/deps/cxxopts/include
)

target_compile_options(measure PRIVATE $<$<COMPILE_LANGUAGE:CXX>:
	# -Wall -Wextra -Wpedantic
    -O3
	${OpenMP_CXX_FLAGS}
	-std=c++14
>)

target_compile_options(measure PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
    --extended-lambda
    --use_fast_math
    --expt-relaxed-constexpr
    -O3
    -Xcompiler=-fopenmp
    -std=c++14
    -gencode arch=compute_75,code=sm_75
    -lineinfo
>)

target_link_libraries(measure
    ${CUDA_LIBRARIES}
    ${CUDA_CUDA_LIBRARY}
    OpenMP::OpenMP_CXX
    cxxopts
)

add_subdirectory(${CMAKE_SOURCE_DIR}/deps/cxxopts)