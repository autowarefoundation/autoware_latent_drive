cmake_minimum_required(VERSION 3.22)
project(latentdrive-trt LANGUAGES CXX)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# ==============================================================================
# Paths
# ==============================================================================
set(SRC_PATH "${CMAKE_CURRENT_SOURCE_DIR}/src")
set(INC_PATH "${CMAKE_CURRENT_SOURCE_DIR}/include")

# CUDA install to link against; override with -DCUDA_DIR=/usr/local/cuda-12.8
# when the default symlink points at a different toolkit than the engine needs
if(NOT CUDA_DIR)
  set(CUDA_DIR "/usr/local/cuda")
endif()

# TensorRT root: -DTENSORRT_ROOT=..., else the TENSORRT_ROOT env var (set by the
# shell rc for a native build), else the system search paths inside the container
if(NOT TENSORRT_ROOT AND DEFINED ENV{TENSORRT_ROOT})
  set(TENSORRT_ROOT $ENV{TENSORRT_ROOT})
endif()

# ==============================================================================
# Collect source files
# ==============================================================================
# Everything except main/ goes into the static library the future ROS 2 node
# links against; main.cpp only carries the gflags mode dispatch
file(GLOB_RECURSE LIB_SRC "${SRC_PATH}/cpp/common/*.cpp"
     "${SRC_PATH}/cpp/latentdrive/*.cpp" "${SRC_PATH}/cpp/visualizer/*.cpp")
file(GLOB_RECURSE MAIN_SRC "${SRC_PATH}/cpp/main/*.cpp")

# ==============================================================================
# Include directories
# ==============================================================================
# Headers are included flat (#include "logger.hpp"), so every subdirectory of
# include/ joins the search path
include_directories(${INC_PATH} ${CUDA_DIR}/include)

file(
  GLOB_RECURSE HEADER_DIRS
  LIST_DIRECTORIES true
  "${INC_PATH}/*")

foreach(HEADER_DIR ${HEADER_DIRS})
  if(IS_DIRECTORY ${HEADER_DIR})
    include_directories(${HEADER_DIR})
  endif()
endforeach()

# ==============================================================================
# Link libraries
# ==============================================================================
find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs videoio)
include_directories(${OpenCV_INCLUDE_DIRS})

# TensorRT ships no CMake config; locate the headers and libraries directly.
# TENSORRT_ROOT can point at a tarball install; the deployment container has it
# in the default system paths. The include dir is put first so a tarball's
# headers win over any older TensorRT installed system-wide
find_path(
  TENSORRT_INCLUDE_DIR NvInfer.h
  HINTS ${TENSORRT_ROOT}
  PATH_SUFFIXES include)
find_library(
  NVINFER_LIB nvinfer
  HINTS ${TENSORRT_ROOT}
  PATH_SUFFIXES lib lib64)
find_library(
  NVONNXPARSER_LIB nvonnxparser
  HINTS ${TENSORRT_ROOT}
  PATH_SUFFIXES lib lib64)
if(NOT TENSORRT_INCLUDE_DIR OR NOT NVINFER_LIB OR NOT NVONNXPARSER_LIB)
  message(FATAL_ERROR "TensorRT not found (set -DTENSORRT_ROOT=/path/to/TensorRT)")
endif()

# Guard against an older TensorRT installed system-wide silently winning when
# TENSORRT_ROOT is unset -- an 8.x header fails deep in the build (no kINT64),
# and linking the wrong major is worse. Fail loudly at configure time instead
file(STRINGS "${TENSORRT_INCLUDE_DIR}/NvInferVersion.h" _trt_major_line
     REGEX "define NV_TENSORRT_MAJOR")
string(REGEX MATCH "[0-9]+" TENSORRT_MAJOR "${_trt_major_line}")
if(TENSORRT_MAJOR LESS 10)
  message(FATAL_ERROR
    "Found TensorRT ${TENSORRT_MAJOR} at ${TENSORRT_INCLUDE_DIR}, but this "
    "project needs >= 10. Point it at a 10.x install with "
    "-DTENSORRT_ROOT=/path/to/TensorRT-10.x (or export TENSORRT_ROOT).")
endif()

message(STATUS "TensorRT ${TENSORRT_MAJOR}: ${TENSORRT_INCLUDE_DIR} | ${NVINFER_LIB}")
include_directories(BEFORE ${TENSORRT_INCLUDE_DIR})

find_library(CUDART_LIB cudart HINTS ${CUDA_DIR}/lib64)

set(LIBS
    ${NVINFER_LIB}
    ${NVONNXPARSER_LIB}
    ${CUDART_LIB}
    gflags
    pthread
    ${OpenCV_LIBS})

# ==============================================================================
# Compiler options
# ==============================================================================
set(CXX_FLAGS "-std=c++17 -pthread -fPIC")

# flags for debug
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
  set(CXX_FLAGS "${CXX_FLAGS} -g -O0")
else()
  set(CXX_FLAGS "${CXX_FLAGS} -O3")
endif()

set(CXX_FLAGS "${CXX_FLAGS} -Wall -Wextra")

foreach(CXX_FILE ${LIB_SRC} ${MAIN_SRC})
  set_source_files_properties(${CXX_FILE} PROPERTIES COMPILE_FLAGS "${CXX_FLAGS}")
endforeach()

# ==============================================================================
# Set library, executable and output path
# ==============================================================================
add_library(latentdrive STATIC ${LIB_SRC})
target_link_libraries(latentdrive PUBLIC ${LIBS})

add_executable(${PROJECT_NAME} ${MAIN_SRC})
target_link_libraries(${PROJECT_NAME} PRIVATE latentdrive)

set_target_properties(
  ${PROJECT_NAME} PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
                             OUTPUT_NAME ${PROJECT_NAME})
