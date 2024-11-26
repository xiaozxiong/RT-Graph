include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

add_library(graph STATIC
    graph.cpp
    thrust_helper.cu
)

target_compile_options(graph PRIVATE $<$<COMPILE_LANGUAGE:CXX>:
	-Wextra
	-Wpedantic
	-O3
>)

target_compile_options(graph PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
	--extended-lambda
	--use_fast_math
	--expt-relaxed-constexpr
	-O3
	-Xcompiler=-fopenmp
	-gencode arch=compute_75,code=sm_75
>)

target_link_libraries(graph
    ${CUDA_LIBRARIES}
    OpenMP::OpenMP_CXX
)


# ====================

add_executable(convertor
    convertor.cpp
)

target_compile_options(convertor PRIVATE
	-O3
)

target_link_libraries(convertor
	graph
)

