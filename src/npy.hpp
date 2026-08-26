// Minimal .npy reader/writer -- float32, C-order only, which is all the
// LatentDrive hand-off package contains. Internal utility for the apps; not
// part of the installed interface.
#pragma once

#include <cstdint>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace latentdrive::npy {

struct Array {
  std::vector<float> data;
  std::vector<std::size_t> shape;

  std::size_t numel() const {
    std::size_t n = 1;
    for (auto s : shape) n *= s;
    return n;
  }
};

inline Array load(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  if (!f) throw std::runtime_error("cannot open " + path);

  char magic[6];
  f.read(magic, 6);
  if (std::string(magic, 6) != "\x93NUMPY") throw std::runtime_error("not npy: " + path);
  uint8_t ver[2];
  f.read(reinterpret_cast<char*>(ver), 2);
  uint32_t header_len = 0;
  if (ver[0] == 1) {
    uint16_t len16;
    f.read(reinterpret_cast<char*>(&len16), 2);
    header_len = len16;
  } else {
    f.read(reinterpret_cast<char*>(&header_len), 4);
  }
  std::string header(header_len, '\0');
  f.read(header.data(), header_len);

  if (header.find("'<f4'") == std::string::npos) {
    throw std::runtime_error(path + ": expected float32 ('<f4'), header: " + header);
  }
  if (header.find("'fortran_order': False") == std::string::npos) {
    throw std::runtime_error(path + ": expected C-order");
  }
  const auto lp = header.find('(');
  const auto rp = header.find(')', lp);
  if (lp == std::string::npos || rp == std::string::npos) {
    throw std::runtime_error(path + ": no shape tuple");
  }

  Array a;
  std::stringstream ss(header.substr(lp + 1, rp - lp - 1));
  std::string tok;
  while (std::getline(ss, tok, ',')) {
    if (tok.find_first_of("0123456789") != std::string::npos) {
      a.shape.push_back(std::stoull(tok));
    }
  }
  a.data.resize(a.numel());
  f.read(reinterpret_cast<char*>(a.data.data()), a.data.size() * sizeof(float));
  if (!f) throw std::runtime_error(path + ": truncated data");
  return a;
}

inline void save(const std::string& path, const float* data,
                 const std::vector<std::size_t>& shape) {
  std::string tup;  // trailing comma is valid tuple syntax for any rank
  for (const auto s : shape) tup += std::to_string(s) + ", ";
  std::string header =
      "{'descr': '<f4', 'fortran_order': False, 'shape': (" + tup + "), }";
  // pad so magic(6)+ver(2)+len(2)+header is a multiple of 64, newline-terminated
  const std::size_t base = 6 + 2 + 2;
  std::size_t total = base + header.size() + 1;
  header.append((64 - total % 64) % 64, ' ');
  header += '\n';

  std::ofstream f(path, std::ios::binary);
  if (!f) throw std::runtime_error("cannot write " + path);
  f.write("\x93NUMPY", 6);
  const uint8_t ver[2] = {1, 0};
  f.write(reinterpret_cast<const char*>(ver), 2);
  const uint16_t len = static_cast<uint16_t>(header.size());
  f.write(reinterpret_cast<const char*>(&len), 2);
  f.write(header.data(), header.size());
  std::size_t n = 1;
  for (auto s : shape) n *= s;
  f.write(reinterpret_cast<const char*>(data), n * sizeof(float));
}

}  // namespace latentdrive::npy
