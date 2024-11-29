include(FetchContent)

# cxxopts
FetchContent_Declare(
	cxxopts
	GIT_REPOSITORY https://github.com/jarro2783/cxxopts.git
	GIT_TAG v3.0.0
	SOURCE_DIR ${CMAKE_SOURCE_DIR}/ext/cxxopts
)
FetchContent_MakeAvailable(cxxopts)

# fmt
# FetchContent_Declare(
# 	fmt
# 	GIT_REPOSITORY https://github.com/fmtlib/fmt.git
# 	GIT_TAG 10.1.1
# 	SOURCE_DIR ${CMAKE_SOURCE_DIR}/ext/fmt
# )
# FetchContent_MakeAvailable(fmt)