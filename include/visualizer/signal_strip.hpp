#ifndef __SIGNAL_STRIP_HPP__
#define __SIGNAL_STRIP_HPP__

#include <memory>
#include <opencv2/core.hpp>
#include <string>
#include <vector>

namespace visualizer {

struct SignalSeries {
  std::string label;
  cv::Scalar  color;
  // A faint series is drawn thin and unlabelled, for the "before" trace that
  // sits under the one that matters
  bool faint = false;
};

struct StripParams {
  std::string title;
  std::string unit;
  double      y_min = 0.0;
  double      y_max = 1.0;
  // Total steps in the clip; fixes the time axis so the trace fills the strip
  // left to right at a constant rate instead of rescaling every frame
  std::size_t n_steps = 1;
};

// @brief `SignalStrip` class is a interface class for a scrolling-free time
// plot: the x axis is the whole clip, samples accumulate behind a playhead,
// and nothing ahead of the playhead is drawn. Reading the raggedness of a
// trace is what makes a stability claim visible -- a plan that pulses shows a
// sawtooth here, a smoothed one does not
class SignalStrip {
 public:
  SignalStrip()          = default;
  virtual ~SignalStrip() = default;

  // @brief: Append one sample per series, in the order the series were
  // declared. Call once per step
  virtual void Push(const std::vector<double> &in_values) = 0;

  // @brief: Render the strip at the given size, playhead on the last sample
  virtual cv::Mat Draw(cv::Size in_size) const = 0;

  virtual void Reset() = 0;
};

// @brief: This function is used as a factory to generate a SignalStrip object
std::shared_ptr<SignalStrip> CreateSignalStrip(const StripParams               &params,
                                               const std::vector<SignalSeries> &series);

}  // namespace visualizer

#endif  //__SIGNAL_STRIP_HPP__
