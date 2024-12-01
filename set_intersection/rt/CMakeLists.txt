include_directories(${OptiX_INCLUDE_DIRS})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
include_directories(${CMAKE_SOURCE_DIR}/utils)

include(${CMAKE_SOURCE_DIR}/cmake/configure_optix.cmake)
cuda_compile_and_embed(embedded_ptx_code device_programs.cu)

set(UILT_CPP ${CMAKE_SOURCE_DIR}/utils/util.cpp)

add_executable(rt_intersection
    main.cpp
    rt_base.cpp
    rt_intersection.cpp
    ${embedded_ptx_code}
	${UILT_CPP}
)

target_compile_options(rt_intersection PRIVATE $<$<COMPILE_LANGUAGE:CXX>:
    -std=c++17
	-O3
>)

target_compile_options(rt_intersection PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
	--extended-lambda
	--use_fast_math
	--expt-relaxed-constexpr
	-O3
	-Xcompiler=-fopenmp
	-gencode arch=compute_75,code=sm_75
	-lineinfo # for profiling
>)

target_link_libraries(rt_intersection
	${CUDA_LIBRARIES}
	${CUDA_CUDA_LIBRARY}
    cxxopts
)
