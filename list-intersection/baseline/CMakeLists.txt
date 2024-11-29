# for all targets added later
include_directories(${CMAKE_SOURCE_DIR}/utils)

set(UILT_CPP ${CMAKE_SOURCE_DIR}/utils/util.cpp)
# 
add_executable(bs bs.cu ${UILT_CPP})
target_link_libraries(bs PRIVATE cxxopts ${CUDA_LIBRARIES} ${CUDA_CUDA_LIBRARY})
target_compile_options(bs
PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:
        -std=c++17
        -O3
        -gencode arch=compute_75,code=sm_75
    >
PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:
        -std=c++17
        -O3
    >
)

add_executable(hi hi.cu ${UILT_CPP})
target_link_libraries(hi PRIVATE cxxopts ${CUDA_LIBRARIES} ${CUDA_CUDA_LIBRARY})
target_compile_options(hi
PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:
        -std=c++17
        -O3
        -gencode arch=compute_75,code=sm_75
    >
PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:
        -std=c++17
        -O3
    >
)

add_executable(bi_naive bi_naive.cu ${UILT_CPP})
target_link_libraries(bi_naive PRIVATE cxxopts ${CUDA_LIBRARIES} ${CUDA_CUDA_LIBRARY})
target_compile_options(bi_naive
PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:
        -std=c++17
        -O3
        -gencode arch=compute_75,code=sm_75
    >
PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:
        -std=c++17
        -O3
    >
)

add_executable(bi_dynamic bi_dynamic.cu ${UILT_CPP})
target_link_libraries(bi_dynamic PRIVATE cxxopts ${CUDA_LIBRARIES} ${CUDA_CUDA_LIBRARY})
target_compile_options(bi_dynamic
PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:
        -std=c++17
        -O3
        -gencode arch=compute_75,code=sm_75
    >
PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:
        -std=c++17
        -O3
    >
)

add_executable(ip ip.cu ${UILT_CPP})
target_link_libraries(ip PRIVATE cxxopts ${CUDA_LIBRARIES} ${CUDA_CUDA_LIBRARY})
target_compile_options(ip
PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:
        -std=c++17
        -O3
        -gencode arch=compute_75,code=sm_75
    >
PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:
        -std=c++17
        -O3
    >
)

# add_executable(set_intersection set_intersection.cu)
# target_link_libraries(set_intersection PRIVATE cxxopts ${CUDA_LIBRARIES} ${CUDA_CUDA_LIBRARY})
# target_compile_options(set_intersection
# PRIVATE
#     $<$<COMPILE_LANGUAGE:CUDA>:
#         -std=c++17
#         -O3
#         -gencode arch=compute_75,code=sm_75
#     >
# PRIVATE
#     $<$<COMPILE_LANGUAGE:CXX>:
#         -std=c++17
#         -O3
#     >
# )