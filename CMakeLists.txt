cmake_minimum_required(VERSION 3.16)
project(IntegrateStreamlines LANGUAGES CXX)

# Add the source directories
set(SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src)
add_subdirectory(${SRC_DIR})

# Set global configurations
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -march=native")
