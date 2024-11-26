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

add_executable(convertor 
    convertor.cpp
)

target_compile_options(convertor PRIVATE
    -O3
)

target_link_libraries(convertor PRIVATE
    graph
)