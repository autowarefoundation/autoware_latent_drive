cmake_minimum_required(VERSION 3.22)
project(latentdrive_ros LANGUAGES CXX)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs)
find_package(CUDAToolkit REQUIRED)

# ============================================================================
# LatentDrive core (the inference/visualizer library, from the parent repo)
# ============================================================================
# Compile the core sources directly rather than depending on the gflags app; the
# repo root is two levels up from this package
set(REPO_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../..")
set(CUDA_DIR "/usr/local/cuda")

if(NOT TENSORRT_ROOT AND DEFINED ENV{TENSORRT_ROOT})
  set(TENSORRT_ROOT $ENV{TENSORRT_ROOT})
endif()
find_path(TENSORRT_INCLUDE_DIR NvInfer.h HINTS ${TENSORRT_ROOT} PATH_SUFFIXES include)
find_library(NVINFER_LIB nvinfer HINTS ${TENSORRT_ROOT} PATH_SUFFIXES lib lib64)
find_library(NVONNXPARSER_LIB nvonnxparser HINTS ${TENSORRT_ROOT} PATH_SUFFIXES lib lib64)
if(NOT TENSORRT_INCLUDE_DIR OR NOT NVINFER_LIB OR NOT NVONNXPARSER_LIB)
  message(FATAL_ERROR "TensorRT not found (set -DTENSORRT_ROOT=/path/to/TensorRT)")
endif()

file(GLOB CORE_SRC
  ${REPO_ROOT}/src/cpp/common/*.cpp
  ${REPO_ROOT}/src/cpp/latentdrive/*.cpp
  ${REPO_ROOT}/src/cpp/visualizer/*.cpp)
add_library(latentdrive_core STATIC ${CORE_SRC})
target_include_directories(latentdrive_core PUBLIC
  ${REPO_ROOT}/include/common
  ${REPO_ROOT}/include/latentdrive
  ${REPO_ROOT}/include/visualizer
  ${REPO_ROOT}/include/utils
  ${TENSORRT_INCLUDE_DIR}
  ${CUDA_DIR}/include)
target_link_libraries(latentdrive_core
  ${NVINFER_LIB} ${NVONNXPARSER_LIB} CUDA::cudart ${OpenCV_LIBS})

# ============================================================================
# ROS 2 node
# ============================================================================
find_package(Threads REQUIRED)
add_executable(latentdrive_node src/latentdrive_node.cpp)
target_link_libraries(latentdrive_node latentdrive_core ${OpenCV_LIBS} Threads::Threads)
ament_target_dependencies(latentdrive_node rclcpp sensor_msgs nav_msgs geometry_msgs tf2_ros)

install(TARGETS latentdrive_node DESTINATION lib/${PROJECT_NAME})
install(DIRECTORY launch config rviz DESTINATION share/${PROJECT_NAME})
install(PROGRAMS scripts/export_route.py scripts/publish_tf_static.py DESTINATION lib/${PROJECT_NAME})

ament_package()
