#include "timer.hpp"

#include <algorithm>

#include "logger.hpp"

namespace timer {

void Timer::StartCpu() {
  c_start_ = std::chrono::high_resolution_clock::now();
}

void Timer::StopCpu(const std::string &msg) {
  const auto   c_stop  = std::chrono::high_resolution_clock::now();
  const double elapsed = std::chrono::duration<double, std::milli>(c_stop - c_start_).count();
  samples_[msg].push_back(elapsed);
}

double Timer::Percentile(const std::string &msg, double percentile) const {
  const auto it = samples_.find(msg);
  if (it == samples_.end() || it->second.empty()) return 0.0;

  std::vector<double> sorted = it->second;
  std::sort(sorted.begin(), sorted.end());
  const double index = percentile / 100.0 * (sorted.size() - 1);
  const auto   lower = static_cast<std::size_t>(index);
  const auto   upper = std::min(lower + 1, sorted.size() - 1);
  return sorted[lower] + (sorted[upper] - sorted[lower]) * (index - lower);
}

double Timer::Last(const std::string &msg) const {
  const auto it = samples_.find(msg);
  return (it == samples_.end() || it->second.empty()) ? 0.0 : it->second.back();
}

int Timer::SampleCount(const std::string &msg) const {
  const auto it = samples_.find(msg);
  return it == samples_.end() ? 0 : static_cast<int>(it->second.size());
}

void Timer::Show() const {
  for (const auto &entry : samples_) {
    LOG("%-24s median %7.2f ms   p99 %7.2f ms   max %7.2f ms   (%d samples)",
        entry.first.c_str(), Percentile(entry.first, 50), Percentile(entry.first, 99),
        Percentile(entry.first, 100), static_cast<int>(entry.second.size()));
  }
}

void Timer::Reset() {
  samples_.clear();
}

};  // namespace timer
