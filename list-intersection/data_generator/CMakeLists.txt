# ===== s_generator =====
add_executable(s_generator
    s_generator.cpp
    data.cpp
)

target_include_directories(s_generator PRIVATE
    ${CMAKE_SOURCE_DIR}/utils
)

target_compile_options(s_generator PRIVATE
    -std=c++17
    -O3
)

target_link_libraries(s_generator PUBLIC
    cxxopts
)
# ===== m_generator =====
add_executable(m_generator
    m_generator.cpp
    data.cpp
)

target_include_directories(m_generator PRIVATE
    ${CMAKE_SOURCE_DIR}/utils
)

target_compile_options(m_generator PRIVATE
    -std=c++17
    -O3
)

target_link_libraries(m_generator PUBLIC
    cxxopts
)

