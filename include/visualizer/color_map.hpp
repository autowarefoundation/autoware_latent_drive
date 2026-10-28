#ifndef __COLOR_MAP_HPP__
#define __COLOR_MAP_HPP__

#include <algorithm>
#include <opencv2/core.hpp>

namespace visualizer {

// @brief: This function is used to map a horizon fraction [0, 1] to a BGR
// color along the green -> yellow -> orange -> red ramp: green marks the
// near/now end of the plan, red the far/+4 s end. Piecewise-linear through a
// small set of control colors, so a 40-waypoint ribbon reads as a smooth
// rainbow of prediction time
inline cv::Scalar HorizonColor(double frac) {
  frac = std::clamp(frac, 0.0, 1.0);
  // control colors in BGR at frac = 0, 1/3, 2/3, 1
  static const cv::Scalar kStops[4] = {
      {80, 220, 40},   // green
      {40, 230, 230},  // yellow
      {40, 150, 245},  // orange
      {48, 48, 235},   // red
  };
  const double scaled = frac * 3.0;
  const int    lo     = std::min(static_cast<int>(scaled), 2);
  const double t      = scaled - lo;
  return kStops[lo] * (1.0 - t) + kStops[lo + 1] * t;
}

}  // namespace visualizer

#endif  //__COLOR_MAP_HPP__
