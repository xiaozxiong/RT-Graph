include_directories(${CMAKE_SOURCE_DIR}/graph/include)

add_executable(linear_bfs
    main.cu
    linear_bfs.cu
)

target_compile_options(linear_bfs PRIVATE $<$<COMPILE_LANGUAGE:CXX>:
	-Wall -Wextra -Wpedantic
>)

target_compile_options(linear_bfs PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
    --extended-lambda
    --use_fast_math
    --expt-relaxed-constexpr
    -O3
    -Xcompiler=-fopenmp
    -gencode arch=compute_75,code=sm_75
    # -lineinfo # for profiling
    # -g -G # enable cuda-gdb
>)

target_link_libraries(linear_bfs
	${CUDA_LIBRARIES}
	${CUDA_CUDA_LIBRARY}
    graph
)