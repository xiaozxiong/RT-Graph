include_directories(${OptiX_INCLUDE_DIRS})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
include_directories(${CMAKE_SOURCE_DIR}/graph/include)

include(${CMAKE_SOURCE_DIR}/cmake/configure_optix.cmake)
cuda_compile_and_embed(embedded_ptx_code device_programs.cu)

add_executable(rt_tc 
	${embedded_ptx_code}
	rt_base.cpp
	cuda_helper.cu
	rt_tc.cpp
	main.cpp
)

target_compile_options(rt_tc PRIVATE $<$<COMPILE_LANGUAGE:CXX>:
	# -Wall 
	# -Wextra 
	# -Wpedantic
	-O3
>)

target_compile_options(rt_tc PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
	--extended-lambda
	--use_fast_math
	--expt-relaxed-constexpr
	-O3
	-Xcompiler=-fopenmp
	-gencode arch=compute_75,code=sm_75
	# -lineinfo # for profiling
>)

target_link_libraries(rt_tc
	${CUDA_LIBRARIES}
	${CUDA_CUDA_LIBRARY}
	graph
)


option(DEBUG "if use log define" OFF)
if(DEBUG)
	target_compile_definitions(rt_tc PRIVATE
		DEBUG
	)
endif()