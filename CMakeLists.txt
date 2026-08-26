cmake_minimum_required(VERSION 3.22)
project(latentdrive_trt LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
add_compile_options(-Wall -Wextra)

find_package(CUDAToolkit REQUIRED)
find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs)

# TensorRT ships no CMake config; locate header + libs directly.
# TENSORRT_ROOT can point at a tarball install; the deployment container has
# them in the default system paths.
find_path(TENSORRT_INCLUDE_DIR NvInfer.h
  HINTS ${TENSORRT_ROOT} PATH_SUFFIXES include)
find_library(TENSORRT_LIB_NVINFER nvinfer
  HINTS ${TENSORRT_ROOT} PATH_SUFFIXES lib lib64)
if(NOT TENSORRT_INCLUDE_DIR OR NOT TENSORRT_LIB_NVINFER)
  message(FATAL_ERROR "TensorRT not found (set -DTENSORRT_ROOT=/path/to/TensorRT)")
endif()
message(STATUS "TensorRT: ${TENSORRT_LIB_NVINFER}")

# --- library: the inference interface a ROS 2 node links against ---
add_library(latentdrive STATIC
  src/planner.cpp
  src/frame_window.cpp
)
target_include_directories(latentdrive
  PUBLIC  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
          $<INSTALL_INTERFACE:include>
  PRIVATE ${TENSORRT_INCLUDE_DIR}
)
target_link_libraries(latentdrive
  PRIVATE ${TENSORRT_LIB_NVINFER} CUDA::cudart
  PUBLIC  opencv_core opencv_imgproc
)

# --- apps ---
add_executable(run_package src/apps/run_package.cpp)
target_include_directories(run_package PRIVATE src)
target_link_libraries(run_package PRIVATE latentdrive opencv_imgcodecs)

install(TARGETS latentdrive run_package)
install(DIRECTORY include/ DESTINATION include)
