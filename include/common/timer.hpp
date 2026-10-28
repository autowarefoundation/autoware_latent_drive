#ifndef __TIMER_HPP__
#define __TIMER_HPP__

#include <chrono>
#include <map>
#include <string>
#include <vector>

namespace timer {

// @brief This customize Timer class. StartCpu/StopCpu pairs accumulate wall
// time samples under a tag, so a benchmark loop can be timed section by
// section (preprocess vs enqueue) and summarized once at the end with the
// percentiles the latency budget cares about
class Timer {
 public:
  Timer()  = default;
  ~Timer() = default;

 public:
  void StartCpu();

  // @brief: This function is used to stop the running clock and store the
  // elapsed time [ms] under `msg`
  void StopCpu(const std::string &msg);

  // @brief: This function is used to get a percentile [0, 100] of the samples
  // stored under `msg`. Returns 0 when the tag has no samples
  double Percentile(const std::string &msg, double percentile) const;

  // @brief: This function is used to get the most recent sample [ms] stored
  // under `msg`, for a live per-frame readout. Returns 0 when empty
  double Last(const std::string &msg) const;

  int SampleCount(const std::string &msg) const;

  // @brief: This function is used to print median/p99/max for every tag
  void Show() const;

  void Reset();

 private:
  std::chrono::time_point<std::chrono::high_resolution_clock> c_start_;
  std::map<std::string, std::vector<double>>                  samples_;
};

};  // namespace timer

#endif  //__TIMER_HPP__
