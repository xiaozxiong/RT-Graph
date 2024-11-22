include_directories(${CMAKE_SOURCE_DIR}/utils)
# include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
# include_directories(${CMAKE_CURRENT_SOURCE_DIR}/mmio)

add_library(graph STATIC
    graph.cpp
    mmio/mmio.cpp
)

target_compile_options(graph PRIVATE $<$<COMPILE_LANGUAGE:CXX>:
    -Wall -Wextra -Wpedantic
>)