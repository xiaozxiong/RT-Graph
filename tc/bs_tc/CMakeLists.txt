include_directories(${CMAKE_SOURCE_DIR}/graph/include)

add_executable(bs_tc
    bs_tc.cu
    main.cu
)

target_compile_options(bs_tc PRIVATE $<$<COMPILE_LANGUAGE:CXX>:
	# -Wall 
	-Wextra
	-Wpedantic
	-O3
>)

target_compile_options(bs_tc PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
	--extended-lambda
	--use_fast_math
	--expt-relaxed-constexpr
	-O3
	-Xcompiler=-fopenmp
	-gencode arch=compute_75,code=sm_75
	# -lineinfo # for profiling
>)

target_link_libraries(bs_tc
	${CUDA_LIBRARIES}
	${CUDA_CUDA_LIBRARY}
    OpenMP::OpenMP_CXX
	graph
)