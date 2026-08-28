#include "signal_strip.hpp"

#include <algorithm>
#include <cmath>
#include <utility>
#include <opencv2/imgproc.hpp>

namespace visualizer {

static const cv::Scalar kBackground{25, 26, 26};
static const cv::Scalar kGrid{42, 44, 44};
static const cv::Scalar kText{183, 194, 195};
static const cv::Scalar kLabel{120, 128, 130};
static const cv::Scalar kPlayhead{80, 86, 88};

constexpr static int kFont = cv::FONT_HERSHEY_SIMPLEX;

// `SignalStripImpl` is an implementation class, which keeps the samples and
// implements the plotting
class SignalStripImpl : public SignalStrip {
 public:
  SignalStripImpl(const StripParams &params, const std::vector<SignalSeries> &series)
      : params_(params), series_(series), samples_(series.size()) {}

  SignalStripImpl(const SignalStripImpl &)            = delete;
  SignalStripImpl &operator=(const SignalStripImpl &) = delete;
  ~SignalStripImpl() override                         = default;

  void Push(const std::vector<double> &in_values) final {
    for (std::size_t i = 0; i < samples_.size(); ++i) {
      samples_[i].push_back(i < in_values.size() ? in_values[i] : 0.0);
    }
  }

  void Reset() final {
    for (auto &s : samples_) s.clear();
  }

  cv::Mat Draw(cv::Size in_size) const final {
    cv::Mat canvas(in_size, CV_8UC3, kBackground);

    // Room on the left for the title block, on the right for the live value
    constexpr int kPadL = 118;
    constexpr int kPadR = 74;
    constexpr int kPadY = 8;
    const int     x0    = kPadL;
    const int     x1    = std::max(in_size.width - kPadR, kPadL + 1);
    const int     y0    = kPadY;
    const int     y1    = std::max(in_size.height - kPadY, kPadY + 1);

    const auto py = [&](double v) {
      const double t = (v - params_.y_min) / std::max(params_.y_max - params_.y_min, 1e-9);
      return static_cast<int>(std::lround(y1 - std::clamp(t, 0.0, 1.0) * (y1 - y0)));
    };
    const auto px = [&](std::size_t k) {
      const double t = static_cast<double>(k) / std::max<std::size_t>(params_.n_steps - 1, 1);
      return static_cast<int>(std::lround(x0 + std::clamp(t, 0.0, 1.0) * (x1 - x0)));
    };

    // Floor, ceiling and midline, labelled at the ends of the value axis
    for (const double v : {params_.y_min, 0.5 * (params_.y_min + params_.y_max), params_.y_max}) {
      cv::line(canvas, {x0, py(v)}, {x1, py(v)}, kGrid, 1);
    }
    // Value-axis ends, right-aligned so they never run into the title block
    for (const auto &end : {std::make_pair(params_.y_max, y0 + 9), std::make_pair(params_.y_min, y1)}) {
      const std::string t = cv::format("%g", end.first);
      const int         w = cv::getTextSize(t, kFont, 0.32, 1, nullptr).width;
      cv::putText(canvas, t, {x0 - w - 6, end.second}, kFont, 0.32, kLabel, 1, cv::LINE_AA);
    }

    // Title block: name, unit, and -- when there is more than one trace -- a
    // swatch per series, all kept out of the plot area
    cv::putText(canvas, params_.title, {6, y0 + 14}, kFont, 0.40, kText, 1, cv::LINE_AA);
    cv::putText(canvas, params_.unit, {6, y0 + 29}, kFont, 0.32, kLabel, 1, cv::LINE_AA);
    if (series_.size() > 1) {
      int ly = y0 + 47;
      for (const SignalSeries &s : series_) {
        cv::line(canvas, {8, ly - 4}, {22, ly - 4}, s.color, s.faint ? 1 : 2, cv::LINE_AA);
        cv::putText(canvas, s.label, {28, ly}, kFont, 0.32, kLabel, 1, cv::LINE_AA);
        ly += 15;
      }
    }

    // Faint series first so the one that matters lands on top
    for (int pass = 0; pass < 2; ++pass) {
      for (std::size_t s = 0; s < series_.size(); ++s) {
        if (series_[s].faint != (pass == 0)) continue;
        const auto &v = samples_[s];
        for (std::size_t k = 0; k + 1 < v.size(); ++k) {
          cv::line(canvas, {px(k), py(v[k])}, {px(k + 1), py(v[k + 1])}, series_[s].color,
                   series_[s].faint ? 1 : 2, cv::LINE_AA);
        }
      }
    }

    if (!samples_.empty() && !samples_[0].empty()) {
      const std::size_t k = samples_[0].size() - 1;
      cv::line(canvas, {px(k), y0}, {px(k), y1}, kPlayhead, 1);

      // Live value of the first non-faint series, at the playhead's height
      int lead = 0;
      for (std::size_t s = 0; s < series_.size(); ++s) {
        if (!series_[s].faint) {
          lead = static_cast<int>(s);
          break;
        }
      }
      const double now = samples_[lead][k];
      cv::circle(canvas, {px(k), py(now)}, 3, series_[lead].color, cv::FILLED, cv::LINE_AA);
      cv::putText(canvas, cv::format("%.1f", now), {x1 + 8, py(now) + 4}, kFont, 0.42,
                  series_[lead].color, 1, cv::LINE_AA);
    }

    return canvas;
  }

 private:
  StripParams                      params_;
  std::vector<SignalSeries>        series_;
  std::vector<std::vector<double>> samples_;
};

std::shared_ptr<SignalStrip> CreateSignalStrip(const StripParams               &params,
                                               const std::vector<SignalSeries> &series) {
  return std::shared_ptr<SignalStrip>(new SignalStripImpl(params, series));
}

}  // namespace visualizer
