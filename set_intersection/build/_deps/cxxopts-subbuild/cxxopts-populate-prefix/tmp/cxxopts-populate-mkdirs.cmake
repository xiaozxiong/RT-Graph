# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file Copyright.txt or https://cmake.org/licensing for details.

cmake_minimum_required(VERSION 3.5)

# If CMAKE_DISABLE_SOURCE_CHANGES is set to true and the source directory is an
# existing directory in our source tree, calling file(MAKE_DIRECTORY) on it
# would cause a fatal error, even though it would be a no-op.
if(NOT EXISTS "/home/xzx/Project/rt-graph/list-intersection/ext/cxxopts")
  file(MAKE_DIRECTORY "/home/xzx/Project/rt-graph/list-intersection/ext/cxxopts")
endif()
file(MAKE_DIRECTORY
  "/home/xzx/Project/rt-graph/list-intersection/build/_deps/cxxopts-build"
  "/home/xzx/Project/rt-graph/list-intersection/build/_deps/cxxopts-subbuild/cxxopts-populate-prefix"
  "/home/xzx/Project/rt-graph/list-intersection/build/_deps/cxxopts-subbuild/cxxopts-populate-prefix/tmp"
  "/home/xzx/Project/rt-graph/list-intersection/build/_deps/cxxopts-subbuild/cxxopts-populate-prefix/src/cxxopts-populate-stamp"
  "/home/xzx/Project/rt-graph/list-intersection/build/_deps/cxxopts-subbuild/cxxopts-populate-prefix/src"
  "/home/xzx/Project/rt-graph/list-intersection/build/_deps/cxxopts-subbuild/cxxopts-populate-prefix/src/cxxopts-populate-stamp"
)

set(configSubDirs )
foreach(subDir IN LISTS configSubDirs)
    file(MAKE_DIRECTORY "/home/xzx/Project/rt-graph/list-intersection/build/_deps/cxxopts-subbuild/cxxopts-populate-prefix/src/cxxopts-populate-stamp/${subDir}")
endforeach()
if(cfgdir)
  file(MAKE_DIRECTORY "/home/xzx/Project/rt-graph/list-intersection/build/_deps/cxxopts-subbuild/cxxopts-populate-prefix/src/cxxopts-populate-stamp${cfgdir}") # cfgdir has leading slash
endif()
