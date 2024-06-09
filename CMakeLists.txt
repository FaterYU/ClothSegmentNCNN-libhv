# The Flutter tooling requires that developers have CMake 3.10 or later
# installed. You should not increase this version, as doing so will cause
# the plugin to fail to compile for some customers of the plugin.
cmake_minimum_required(VERSION 3.10)

set(CMAKE_CXX_STANDARD 11)

project(yolo8_ncnn VERSION 0.0.1 LANGUAGES C CXX)

# set current project directory
set(MY_PROJ "${CMAKE_CURRENT_SOURCE_DIR}")
message(STATUS "MY_PROJ: ${MY_PROJ}")

find_package(libhv REQUIRED)

# add opencv
find_package(OpenCV REQUIRED)
message(STATUS "OpenCV_FOUND: ${OpenCV_FOUND}")
message(STATUS "OpenCV_VERSION: ${OpenCV_VERSION}")
message(STATUS "OpenCV_INCLUDE_DIRS: ${OpenCV_INCLUDE_DIRS}")
message(STATUS "OpenCV_LIBS: ${OpenCV_LIBS}")

include_directories(
    ${MY_PROJ}/src
    ${OpenCV_INCLUDE_DIRS}
)

set(ncnn_DIR "${MY_PROJ}/lib/ncnn-20240102-ubuntu-2204-shared/lib/cmake/ncnn")

if(NOT EXISTS ${ncnn_DIR})
    message(FATAL_ERROR "ncnn_DIR not exists: ${ncnn_DIR}")
endif()

# # ncnn
find_package(ncnn REQUIRED)
message(STATUS "ncnn_FOUND: ${ncnn_FOUND}")

link_directories(${CMAKE_BINARY_DIR}/bin)
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_SOURCE_DIR}/bin)

# targets
add_executable(yolo8_ncnn "src/server.cpp")
target_link_libraries(yolo8_ncnn hv ncnn ${OpenCV_LIBS})
