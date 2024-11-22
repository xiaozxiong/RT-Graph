include_directories(${OptiX_INCLUDE_DIRS})
include_directories(${PROJECT_SOURCE_DIR}/graph/include)

include(${PROJECT_SOURCE_DIR}/cmake/configure_optix.cmake)
cuda_compile_and_embed(embedded_ptx_code device_programs.cu)

add_executable(rt_bfs 
    ${embedded_ptx_code}
    main.cpp
	rt_bfs.cu
	cuda_helper.cu
	record.h
)

target_compile_options(rt_bfs PRIVATE
	$<$<COMPILE_LANGUAGE:CXX>:
	# -Wall -Wextra -Wpedantic
	-O3
	# ${OpenMP_CXX_FLAGS}
>)

target_compile_options(rt_bfs PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
	--extended-lambda
	--use_fast_math
	--expt-relaxed-constexpr
	-O3
	-Xcompiler=-fopenmp
	-gencode arch=compute_75,code=sm_75
	# -lineinfo # for profiling
>)

target_link_libraries(rt_bfs
	${CUDA_LIBRARIES} # 包含了所有 CUDA 库的名称
	${CUDA_CUDA_LIBRARY} # 包含了 CUDA 的运行时库的名称
	OpenMP::OpenMP_CXX
    graph
)


# =============== v2
cuda_compile_and_embed(embedded_ptx_code_v2 device_programs_v2.cu)

add_executable(rt_bfs_v2 
    ${embedded_ptx_code_v2}
	record.h
	rt_bfs_v2.cu
	cuda_helper.cu
	main_v2.cpp
)

target_compile_options(rt_bfs_v2 PRIVATE
	$<$<COMPILE_LANGUAGE:CXX>:
	# -Wall -Wextra -Wpedantic
	# ${OpenMP_CXX_FLAGS}
	-O3
>)

target_compile_options(rt_bfs_v2 PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
	--extended-lambda
	--use_fast_math
	--expt-relaxed-constexpr
	-O3
	-Xcompiler=-fopenmp
	-gencode arch=compute_75,code=sm_75
	# -lineinfo # for profiling
>)

target_link_libraries(rt_bfs_v2
	${CUDA_LIBRARIES} # 包含了所有 CUDA 库的名称
	${CUDA_CUDA_LIBRARY} # 包含了 CUDA 的运行时库的名称
	OpenMP::OpenMP_CXX
    graph
)