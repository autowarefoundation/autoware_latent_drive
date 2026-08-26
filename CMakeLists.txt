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
set(CUDA_DIR "/usr/local/cuda")

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

# TensorRT ships no CMake config; locate the library directly. TENSORRT_ROOT
# can point at a tarball install; the deployment container has it in the
# default system paths
find_library(
  NVINFER_LIB nvinfer
  HINTS ${TENSORRT_ROOT}
  PATH_SUFFIXES lib lib64)
if(NOT NVINFER_LIB)
  message(FATAL_ERROR "TensorRT not found (set -DTENSORRT_ROOT=/path/to/TensorRT)")
endif()

find_library(CUDART_LIB cudart HINTS ${CUDA_DIR}/lib64)

set(LIBS
    ${NVINFER_LIB}
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
