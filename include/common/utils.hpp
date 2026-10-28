#ifndef __UTILS_HPP__
#define __UTILS_HPP__

#include <cuda_runtime.h>

#include <fstream>
#include <memory>
#include <string>
#include <vector>

#include "logger.hpp"

#define CUDA_CHECK(call) __cudaCheck(call, __FILE__, __LINE__)

static void __cudaCheck(cudaError_t err, const char *file, const int line) {
  if (err != cudaSuccess) {
    printf("ERROR: %s:%d, ", file, line);
    printf("code:%s, reason:%s\n", cudaGetErrorName(err), cudaGetErrorString(err));
    exit(1);
  }
}

namespace common {

// @brief: This function is used to read a whole binary file (engine plan,
// calibration cache, ...) into memory. Returns an empty vector on failure
inline std::vector<char> LoadFile(const std::string &path) {
  std::ifstream file(path, std::ios::binary | std::ios::ate);
  if (!file) {
    LOGE("Failed to open %s", path.c_str());
    return {};
  }
  const std::streamsize size = file.tellg();
  file.seekg(0);
  std::vector<char> data(static_cast<std::size_t>(size));
  if (!file.read(data.data(), size)) {
    LOGE("Short read on %s", path.c_str());
    return {};
  }
  return data;
}

namespace cuda {

struct Deleter {
  void operator()(void *p) const { CUDA_CHECK(::cudaFree(p)); }
};

struct HostDeleter {
  void operator()(void *p) const { CUDA_CHECK(::cudaFreeHost(p)); }
};

// unique pointer for cuda device memory
template <typename T>
using cuda_unique_ptr = std::unique_ptr<T, Deleter>;

// unique pointer for cuda pinned host memory
template <typename T>
using cuda_host_unique_ptr = std::unique_ptr<T, HostDeleter>;

// unique pointer for TensorRT interface classes (destroy() is deprecated
// since TensorRT 8, plain delete is the supported release path)
template <typename T>
using nv_unique_ptr = std::unique_ptr<T>;

// @brief: This function is used to allocate `size` bytes of device memory
template <typename T>
cuda_unique_ptr<T> MakeDeviceMemory(std::size_t size) {
  void *ptr = nullptr;
  CUDA_CHECK(::cudaMalloc(&ptr, size));
  return cuda_unique_ptr<T>(static_cast<T *>(ptr));
}

// @brief: This function is used to allocate `size` bytes of pinned host memory
template <typename T>
cuda_host_unique_ptr<T> MakeHostMemory(std::size_t size) {
  void *ptr = nullptr;
  CUDA_CHECK(::cudaMallocHost(&ptr, size));
  return cuda_host_unique_ptr<T>(static_cast<T *>(ptr));
}

}  // namespace cuda
}  // namespace common

#endif  //__UTILS_HPP__
