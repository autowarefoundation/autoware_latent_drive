#ifndef __NPY_HPP__
#define __NPY_HPP__

#include <cstdint>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "logger.hpp"

namespace utils {

// Minimal .npy support -- float32, C-order only, which is all the LatentDrive
// hand-off package contains. Errors are fatal: these are offline tools and a
// broken input file has no recovery path

struct NpyArray {
  std::vector<float>       data;
  std::vector<std::size_t> shape;

  std::size_t Numel() const {
    std::size_t n = 1;
    for (const auto s : shape) n *= s;
    return n;
  }
};

// @brief: This function is used to load a float32 C-order .npy file
inline NpyArray LoadNpy(const std::string &path) {
  std::ifstream file(path, std::ios::binary);
  if (!file) LOGF("Cannot open %s", path.c_str());

  char magic[6];
  file.read(magic, 6);
  if (std::string(magic, 6) != "\x93NUMPY") LOGF("Not a npy file: %s", path.c_str());

  uint8_t version[2];
  file.read(reinterpret_cast<char *>(version), 2);
  uint32_t header_len = 0;
  if (version[0] == 1) {
    uint16_t header_len16;
    file.read(reinterpret_cast<char *>(&header_len16), 2);
    header_len = header_len16;
  } else {
    file.read(reinterpret_cast<char *>(&header_len), 4);
  }
  std::string header(header_len, '\0');
  file.read(header.data(), header_len);

  if (header.find("'<f4'") == std::string::npos) {
    LOGF("%s: expected float32 ('<f4'), header: %s", path.c_str(), header.c_str());
  }
  if (header.find("'fortran_order': False") == std::string::npos) {
    LOGF("%s: expected C-order", path.c_str());
  }
  const auto lparen = header.find('(');
  const auto rparen = header.find(')', lparen);
  if (lparen == std::string::npos || rparen == std::string::npos) {
    LOGF("%s: no shape tuple in header", path.c_str());
  }

  NpyArray          array;
  std::stringstream shape_ss(header.substr(lparen + 1, rparen - lparen - 1));
  std::string       token;
  while (std::getline(shape_ss, token, ',')) {
    if (token.find_first_of("0123456789") != std::string::npos) {
      array.shape.push_back(std::stoull(token));
    }
  }
  array.data.resize(array.Numel());
  file.read(reinterpret_cast<char *>(array.data.data()), array.data.size() * sizeof(float));
  if (!file) LOGF("%s: truncated data section", path.c_str());
  return array;
}

// @brief: This function is used to save a float32 C-order .npy file
inline void SaveNpy(const std::string &path, const float *data,
                    const std::vector<std::size_t> &shape) {
  // A trailing comma is valid tuple syntax for any rank, so always append one
  std::string shape_tuple;
  for (const auto s : shape) shape_tuple += std::to_string(s) + ", ";

  std::string header =
      "{'descr': '<f4', 'fortran_order': False, 'shape': (" + shape_tuple + "), }";
  // Pad so magic(6) + version(2) + length(2) + header is 64-aligned
  const std::size_t unpadded = 6 + 2 + 2 + header.size() + 1;
  header.append((64 - unpadded % 64) % 64, ' ');
  header += '\n';

  std::ofstream file(path, std::ios::binary);
  if (!file) LOGF("Cannot write %s", path.c_str());

  const uint8_t  version[2] = {1, 0};
  const uint16_t header_len = static_cast<uint16_t>(header.size());
  file.write("\x93NUMPY", 6);
  file.write(reinterpret_cast<const char *>(version), 2);
  file.write(reinterpret_cast<const char *>(&header_len), 2);
  file.write(header.data(), header.size());

  std::size_t numel = 1;
  for (const auto s : shape) numel *= s;
  file.write(reinterpret_cast<const char *>(data), numel * sizeof(float));
}

}  // namespace utils

#endif  //__NPY_HPP__
