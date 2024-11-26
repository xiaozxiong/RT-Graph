cmake_minimum_required(VERSION 3.20)

if(NOT DEFINED CMAKE_CUDA_COMPILER)
    set(CMAKE_CUDA_COMPILER /usr/local/cuda-11.8/bin/nvcc)
endif()

project(TC C CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_MODULE_PATH
	${CMAKE_MODULE_PATH}
    "${PROJECT_SOURCE_DIR}/cmake"
)
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_SOURCE_DIR}/bin)
set(LIBRARY_OUTPUT_PATH ${CMAKE_SOURCE_DIR}/lib)

find_package(CUDA REQUIRED)
if(CUDA_FOUND)
	include_directories(${CUDA_INCLUDE_DIRS})
	message(STATUS "CUDA_INCLUDE_DIRS: " ${CUDA_INCLUDE_DIRS})
else()
	message(FATAL_ERROR "CUDA Not Found")
endif()

find_package(OptiX REQUIRED)
if(OptiX_FOUND)
	message(STATUS "OptiX_INCLUDE_DIRS: " ${OptiX_INCLUDE_DIRS})
	# don't include here
else()
	message(FATAL_ERROR "OptiX Not Found")
endif()

find_package(OpenMP REQUIRED)

include_directories(${CMAKE_SOURCE_DIR}/utils)
# subdirectory
add_subdirectory(${CMAKE_SOURCE_DIR}/graph)
add_subdirectory(${CMAKE_SOURCE_DIR}/rt_tc)
add_subdirectory(${CMAKE_SOURCE_DIR}/bs_tc)



