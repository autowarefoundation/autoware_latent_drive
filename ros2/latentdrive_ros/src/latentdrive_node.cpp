// Minimal ROS 2 node for open-loop rosbag replay of the LatentDrive planner.
//
// Subscribes to the cameras, ego odometry and acceleration; every view-0
// (CAM_FRONT) frame triggers a plan: the timestamped window picks the four
// history slots, the status builder makes the status vector, the engine runs,
// and the node publishes the plan plus a camera+BEV visualization image.
//
// The engine decides the contract, the node adapts:
//   views         1 (CAM_FRONT), 2 (wide + telephoto) or 3 (camfront3: front,
//                 front_left, front_right) -- the first NumViews() entries of
//                 `camera_topics`
//   status        [1, 6] (one subgoal 50 m ahead) or [1, 16] (six route
//                 points at 5, 10, 15, 20, 30, 50 m)
//   waypoints     8 at 2 Hz or 40 at 10 Hz
//
// Cameras: view 0 anchors the history (nearest stamp to t-1.5, t-1.0, t-0.5,
// t); every other view supplies, per slot, its frame nearest in time to the
// chosen view-0 frame, within `view_tolerance_s` (50 ms, as the camfront3
// training paired them). The cameras are not synchronised, so the side frame
// belonging to the newest front frame may still be in flight when the front
// frame arrives: the plan waits for it, up to the tolerance past the front
// stamp. A slot that stays unmatched skips the inference, or with
// `missing_view_fill:=zeros` gets the blank frame the training fed for a
// missing view. Each camera decodes on its own thread (callback group), so
// three 2880x1860 JPEG decodes overlap instead of queueing inside one 100 ms
// period, and the inference runs on a planning thread of its own: a camera
// callback only ever waits for a frame push, never for an engine call (the
// vehicle node's decode thread + tick thread split).
//
// Tick: one plan per view-0 frame, paced on the frame header stamps to at
// most `max_rate_hz` (the vehicle node's `tick_trigger:=camera`). Arrival
// times jitter by tens of milliseconds under bag playback; the stamps do not.
//
// Route (the status' route points), `route_source`:
//   recorded  the route points the vehicle's planner fed the model, recorded
//             in the bag on /tensorrt_e2e/debug/latentdrive/route_points (map
//             frame) -- the same input the car saw, from its lanelet route
//   file      a map-frame polyline from `route_file` (route.npy, e.g. the
//             ego's driven path from scripts/export_route.py), turned into
//             route points the way the vehicle does it (1 m resample, forward-
//             hinted projection, clamp at the end)
// Motion terms (`motion_source`): pose_difference (the training convention,
// differenced positions 0.1 s apart) or twist (the odometry twist and the
// acceleration topic).
#include <algorithm>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstddef>
#include <cstring>
#include <deque>
#include <fstream>
#include <memory>
#include <mutex>
#include <optional>
#include <pthread.h>
#include <string>
#include <thread>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <nlohmann/json.hpp>
#include <rclcpp/rclcpp.hpp>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>

#include <geometry_msgs/msg/accel_with_covariance_stamped.hpp>
#include <geometry_msgs/msg/pose_array.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <nav_msgs/msg/path.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <sensor_msgs/msg/compressed_image.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <std_msgs/msg/header.hpp>

#include "bev_visualizer.hpp"
#include "camera_projector.hpp"
#include "color_map.hpp"
#include "frame_window.hpp"
#include "npy.hpp"
#include "planner.hpp"
#include "status_builder.hpp"
#include "trajectory.hpp"
#include "trajectory_smoother.hpp"

using namespace std::chrono_literals;

namespace {

// @brief: This function is used to load a scene entry from a calibration.json
// (K_crop_resized for the 512x256 image and the camera-to-ego extrinsics) of
// the hand-off package, returning a projector for the camera overlay
std::shared_ptr<visualizer::CameraProjector> LoadProjector(const std::string &path,
                                                           const std::string &match) {
  std::ifstream f(path);
  if (!f) return nullptr;
  const auto cal = nlohmann::json::parse(f);
  for (auto it = cal["scenes"].begin(); it != cal["scenes"].end(); ++it) {
    if (it.key().find(match) == std::string::npos) continue;
    const auto &e = it.value();
    visualizer::CameraParams p;
    for (int r = 0; r < 3; ++r) {
      for (int c = 0; c < 3; ++c) {
        p.k(r, c) = e["K_crop_resized"][r][c].get<double>();
        p.r(r, c) = e["extrinsics_camera_to_ego"]["rotation_matrix"][r][c].get<double>();
      }
      p.t[r] = e["extrinsics_camera_to_ego"]["translation"][r].get<double>();
    }
    return visualizer::CreateCameraProjector(p);
  }
  return nullptr;
}

double YawOf(const geometry_msgs::msg::Quaternion &q) {
  return std::atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z));
}

// @brief: The vehicle node's camera-trigger pacing: plan on this frame when at
// least one period (minus half a frame gap, so a 10 Hz camera is not beaten by
// its own jitter) has passed since the last planned frame's stamp. True on
// the first frame and when the clock ran backwards (a looping bag)
bool TriggerDue(double stamp_s, std::optional<double> last_tick_s, double gap_s, double period_s) {
  if (!last_tick_s || stamp_s < *last_tick_s) return true;
  return stamp_s - *last_tick_s >= period_s - 0.5 * gap_s;
}

}  // namespace

class LatentDriveNode : public rclcpp::Node {
 public:
  LatentDriveNode() : Node("latentdrive_node") {
    const auto engine = declare_parameter<std::string>("engine_file", "");
    const auto calib = declare_parameter<std::string>("calibration_file", "");
    const auto scene_match = declare_parameter<std::string>("scene_match", "prd_jt");
    const auto smooth_alpha = declare_parameter<double>("smooth_alpha", 0.35);
    viz_jpeg_quality_ = static_cast<int>(declare_parameter<int64_t>("viz_jpeg_quality", 80));
    const auto viz_reliable = declare_parameter<bool>("viz_reliable", false);
    // View order of the camfront3 models: front, front_left, front_right. On
    // the prd_jt vehicles these are camera0 (57 deg), camera3 (+49.5 deg) and
    // camera4 (-49.6 deg); a single-view engine reads the first one only
    camera_topics_ = declare_parameter<std::vector<std::string>>(
        "camera_topics", std::vector<std::string>{"/sensing/camera/camera0/image_raw/compressed",
                                                  "/sensing/camera/camera3/image_raw/compressed",
                                                  "/sensing/camera/camera4/image_raw/compressed"});
    view_tolerance_s_ = declare_parameter<double>("view_tolerance_s",
                                                  model::planner::kDefaultViewToleranceS);
    const auto history_tolerance_s = declare_parameter<double>(
        "history_tolerance_s", model::planner::kDefaultHistoryToleranceS);
    const auto missing_view_fill = declare_parameter<std::string>("missing_view_fill", "skip");
    const auto max_rate_hz = declare_parameter<double>("max_rate_hz", 10.0);
    const auto odom_topic = declare_parameter<std::string>(
        "odom_topic", "/localization/kinematic_state");
    const auto accel_topic = declare_parameter<std::string>(
        "accel_topic", "/localization/acceleration");
    route_source_ = declare_parameter<std::string>("route_source", "recorded");
    const auto route = declare_parameter<std::string>("route_file", "");
    const auto recorded_topic = declare_parameter<std::string>(
        "recorded_route_points_topic", "/tensorrt_e2e/debug/latentdrive/route_points");
    const auto route_points_m =
        declare_parameter<std::vector<double>>("route_points_m", std::vector<double>{});
    const auto route_reversal_deg = declare_parameter<double>(
        "route_reversal_deg", model::planner::kDefaultRouteReversalDeg);
    const auto motion_source = declare_parameter<std::string>("motion_source", "pose_difference");

    if (missing_view_fill != "skip" && missing_view_fill != "zeros") {
      throw std::runtime_error("missing_view_fill must be skip or zeros, got " + missing_view_fill);
    }
    if (route_source_ != "recorded" && route_source_ != "file") {
      throw std::runtime_error("route_source must be recorded or file, got " + route_source_);
    }
    if (motion_source != "pose_difference" && motion_source != "twist") {
      throw std::runtime_error("motion_source must be pose_difference or twist, got " + motion_source);
    }
    period_s_ = max_rate_hz > 0.0 ? 1.0 / max_rate_hz : 0.0;

    planner_ = model::planner::CreatePlanner(engine);
    if (!planner_) throw std::runtime_error("failed to load the engine " + engine);
    num_views_ = planner_->NumViews();
    if (num_views_ > static_cast<int>(camera_topics_.size())) {
      throw std::runtime_error("engine takes " + std::to_string(num_views_) + " views but camera_topics lists " +
                               std::to_string(camera_topics_.size()));
    }
    camera_topics_.resize(static_cast<std::size_t>(num_views_));

    preprocess::ViewWindowParams window_params;
    window_params.num_views = num_views_;
    window_params.view_tolerance_s = view_tolerance_s_;
    window_params.history_tolerance_s = history_tolerance_s;
    window_params.fill_missing_views = missing_view_fill == "zeros";
    window_ = preprocess::CreateViewWindow(window_params);

    status::StatusBuilderParams status_params;
    status_params.num_route_points = planner_->NumRoutePoints();
    status_params.route_points_m = route_points_m;
    status_params.route_reversal_deg = route_reversal_deg;
    status_params.motion_source = motion_source == "twist" ? status::MotionSource::kTwist
                                                           : status::MotionSource::kPoseDifference;
    status_ = status::CreateStatusBuilder(status_params);

    bev_ = visualizer::CreateBevVisualizer();
    // Display only: ~/plan carries the raw engine output, the overlay is what
    // gets filtered. Consecutive plans disagree by about a metre on how far
    // they reach, which makes the BEV tail pulse in and out
    smoother_ = visualizer::CreateTrajectorySmoother({smooth_alpha});
    if (!window_ || !status_ || !bev_) {
      throw std::runtime_error("failed to init the window / status builder / bev");
    }
    if (!calib.empty()) {
      projector_ = LoadProjector(calib, scene_match);
      if (!projector_) {
        RCLCPP_WARN(get_logger(), "no calibration entry matching '%s' in %s", scene_match.c_str(),
                    calib.c_str());
      }
    }
    if (route_source_ == "file") LoadRoute(route);

    // warmup so the first real inference is not slow
    const std::vector<float> zeros(planner_->VideoNumel(), 0.f);
    const std::vector<float> zero_status(static_cast<std::size_t>(planner_->StatusDim()), 0.f);
    for (int i = 0; i < 5; ++i) planner_->Forward(zeros.data(), zero_status);

    // Each camera gets its own callback group, spun by its own single-threaded
    // executor on its own thread: the JPEG decodes and the preprocessing of
    // the cameras overlap instead of serialising inside one 100 ms period,
    // and the frames of one camera stay in order. Not one MultiThreadedExecutor
    // over mutually exclusive groups: while one camera's callback runs, its
    // subscription keeps the other executor threads waking up for work they
    // cannot take, and on humble that spin cost four cores and left the
    // callbacks seconds behind the bag. What the callbacks share with the
    // planning thread -- the window, the pending plan, the status builder --
    // sits behind plan_mutex_
    const rclcpp::QoS sensor_qos = rclcpp::SensorDataQoS();
    // Best effort like the camera drivers, but a deeper queue than the
    // sensor profile's 5: a bag player that falls behind on a heavy stretch
    // catches up in bursts, and a burst longer than the queue loses side
    // frames -- which skips plans, as the front frame then has no partner
    const rclcpp::QoS camera_qos = rclcpp::SensorDataQoS().keep_last(30);
    for (int v = 0; v < num_views_; ++v) {
      auto group = create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive, false);
      rclcpp::SubscriptionOptions opts;
      opts.callback_group = group;
      sub_images_.push_back(create_subscription<sensor_msgs::msg::CompressedImage>(
          camera_topics_[static_cast<std::size_t>(v)], camera_qos,
          [this, v](sensor_msgs::msg::CompressedImage::ConstSharedPtr msg) { OnImage(v, msg); },
          opts));
      auto executor = std::make_shared<rclcpp::executors::SingleThreadedExecutor>();
      executor->add_callback_group(group, get_node_base_interface());
      camera_executors_.push_back(executor);
    }
    sub_odom_ = create_subscription<nav_msgs::msg::Odometry>(
        odom_topic, sensor_qos, [this](nav_msgs::msg::Odometry::ConstSharedPtr m) { OnOdometry(m); });
    sub_accel_ = create_subscription<geometry_msgs::msg::AccelWithCovarianceStamped>(
        accel_topic, sensor_qos,
        [this](geometry_msgs::msg::AccelWithCovarianceStamped::ConstSharedPtr m) {
          std::lock_guard<std::mutex> lock(plan_mutex_);
          accel_ = m;
        });
    if (route_source_ == "recorded") {
      sub_recorded_ = create_subscription<geometry_msgs::msg::PoseArray>(
          recorded_topic, rclcpp::QoS(10),
          [this](geometry_msgs::msg::PoseArray::ConstSharedPtr m) { OnRecordedRoutePoints(m); });
    }

    // Without a calibration file the overlay comes from the bag itself: the
    // view-0 camera_info (intrinsics) and /tf_static (base_link -> camera)
    if (!projector_) {
      tf_buffer_ = std::make_shared<tf2_ros::Buffer>(get_clock());
      tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);
      std::string info_topic = camera_topics_[0];
      for (const std::string suffix : {"/image_raw/compressed", "/image_rect_color/compressed"}) {
        const auto pos = info_topic.rfind(suffix);
        if (pos != std::string::npos && pos + suffix.size() == info_topic.size()) {
          info_topic = info_topic.substr(0, pos) + "/camera_info";
        }
      }
      sub_info_ = create_subscription<sensor_msgs::msg::CameraInfo>(
          info_topic, sensor_qos, [this](sensor_msgs::msg::CameraInfo::ConstSharedPtr m) {
            std::lock_guard<std::mutex> lock(viz_mutex_);
            camera_info_ = m;
          });
    }

    // Sensor QoS on the viz topics: best-effort, keep-last. A live overlay must
    // never apply back-pressure -- with the default reliable, depth-1 profile a
    // viewer that falls behind blocks the publisher, and the stall reaches the
    // image callback as dropped camera frames (measured: 5.7 Hz with a viewer
    // attached and four "stamp jumped" resets over a 60 s bag)
    const rclcpp::QoS viz_qos = viz_reliable ? rclcpp::QoS(1).reliable() : sensor_qos;
    pub_viz_ = create_publisher<sensor_msgs::msg::Image>("~/viz", viz_qos);
    pub_viz_jpeg_ =
        create_publisher<sensor_msgs::msg::CompressedImage>("~/viz/compressed", viz_qos);
    pub_plan_ = create_publisher<nav_msgs::msg::Path>("~/plan", 1);
    pub_plan_map_ = create_publisher<nav_msgs::msg::Path>("~/plan_map", 1);
    pub_route_points_ = create_publisher<geometry_msgs::msg::PoseArray>("~/route_points", 1);

    // Named, so `top -H` tells the cameras, the planner and the drawing apart
    viz_thread_ = std::thread(&LatentDriveNode::VizLoop, this);
    pthread_setname_np(viz_thread_.native_handle(), "ld_viz");
    plan_thread_ = std::thread(&LatentDriveNode::PlanLoop, this);
    pthread_setname_np(plan_thread_.native_handle(), "ld_plan");
    for (std::size_t v = 0; v < camera_executors_.size(); ++v) {
      auto executor = camera_executors_[v];
      camera_threads_.emplace_back([executor] { executor->spin(); });
      pthread_setname_np(camera_threads_.back().native_handle(), ("ld_camera" + std::to_string(v)).c_str());
    }

    std::string views;
    for (const auto &t : camera_topics_) views += (views.empty() ? "" : ", ") + t;
    RCLCPP_INFO(get_logger(),
                "latentdrive_node ready: %d view%s [%s] (side views within %.0f ms, missing -> %s), "
                "status [1, %d] = %d route point%s from %s, motion from %s, %d waypoints, "
                "at most %.1f Hz on the view-0 stamps",
                num_views_, num_views_ == 1 ? "" : "s", views.c_str(), view_tolerance_s_ * 1e3,
                missing_view_fill.c_str(), planner_->StatusDim(), planner_->NumRoutePoints(),
                planner_->NumRoutePoints() == 1 ? "" : "s",
                route_source_ == "recorded" ? recorded_topic.c_str() : route.c_str(),
                motion_source.c_str(), planner_->NumWaypoints(), max_rate_hz);
  }

  ~LatentDriveNode() override {
    for (auto &executor : camera_executors_) executor->cancel();
    for (auto &thread : camera_threads_) {
      if (thread.joinable()) thread.join();
    }
    {
      std::lock_guard<std::mutex> lock(plan_mutex_);
      plan_stop_ = true;
    }
    plan_cv_.notify_one();
    if (plan_thread_.joinable()) plan_thread_.join();
    {
      std::lock_guard<std::mutex> lock(viz_mutex_);
      viz_stop_ = true;
    }
    viz_cv_.notify_one();
    if (viz_thread_.joinable()) viz_thread_.join();
  }

 private:
  // The view-0 frame waiting to be planned on, with the ego sampled at its
  // arrival; the status is filled in when the plan is staged
  struct Pending {
    double stamp_s;
    builtin_interfaces::msg::Time stamp;
    cv::Mat frame;
    status::StatusResult status;
    status::PoseSample ego;
    double ego_z = 0.0;
  };

  // A staged plan: its assembled [V, 3, 4, H, W] window and how many of its
  // side slots carry the blank frame
  struct ReadyJob {
    Pending pending;
    std::vector<float> video;
    std::size_t filled = 0;
  };

  // One in-flight drawing job, latest wins
  struct VizJob {
    cv::Mat frame;  // the decoded view-0 frame, full resolution
    common::Trajectory plan;
    common::EgoStatus ego;
    std::size_t filled = 0;
  };

  // @brief: route.npy is [N, 2] or [N, 3] (map x, y[, yaw]) in driving order
  void LoadRoute(const std::string &path) {
    if (path.empty()) throw std::runtime_error("route_source:=file needs route_file");
    const auto a = utils::LoadNpy(path);
    if (a.shape.size() != 2 || (a.shape[1] != 2 && a.shape[1] != 3) || a.shape[0] < 2) {
      throw std::runtime_error(path + " is not an [N, 2|3] route");
    }
    std::vector<status::Point2d> route(a.shape[0]);
    for (std::size_t i = 0; i < a.shape[0]; ++i) {
      route[i] = {a.data[i * a.shape[1] + 0], a.data[i * a.shape[1] + 1]};
    }
    status_->SetRoute(route);
    RCLCPP_INFO(get_logger(), "loaded route: %zu points from %s", route.size(), path.c_str());
  }

  void OnOdometry(nav_msgs::msg::Odometry::ConstSharedPtr m) {
    std::lock_guard<std::mutex> lock(plan_mutex_);
    odom_ = m;
    status_->PushPose(ToSample(*m));
  }

  status::PoseSample ToSample(const nav_msgs::msg::Odometry &m) const {
    status::PoseSample s;
    s.stamp_s = rclcpp::Time(m.header.stamp).seconds();
    s.x = m.pose.pose.position.x;
    s.y = m.pose.pose.position.y;
    s.yaw = YawOf(m.pose.pose.orientation);
    s.vel_x = m.twist.twist.linear.x;
    s.vel_y = m.twist.twist.linear.y;
    if (accel_) {
      s.acc_x = accel_->accel.accel.linear.x;
      s.acc_y = accel_->accel.accel.linear.y;
    }
    return s;
  }

  void OnRecordedRoutePoints(geometry_msgs::msg::PoseArray::ConstSharedPtr m) {
    std::lock_guard<std::mutex> lock(plan_mutex_);
    const double stamp_s = rclcpp::Time(m->header.stamp).seconds();
    if (!recorded_.empty() && stamp_s < recorded_.back().first - 1.0) recorded_.clear();
    std::vector<status::Point2d> points;
    for (const auto &p : m->poses) points.push_back({p.position.x, p.position.y});
    recorded_.emplace_back(stamp_s, std::move(points));
    while (recorded_.size() > 50) recorded_.pop_front();
  }

  // @brief: the recorded route points nearest in time to `stamp_s`. The
  // vehicle stamped them with its tick, a few ms after the camera frame it
  // planned on; more than kRecordedMaxGapS away and they belong to another tick
  std::optional<std::vector<status::Point2d>> RecordedAt(double stamp_s) const {
    const std::pair<double, std::vector<status::Point2d>> *best = nullptr;
    for (const auto &r : recorded_) {
      if (!best || std::abs(r.first - stamp_s) < std::abs(best->first - stamp_s)) best = &r;
    }
    if (!best || std::abs(best->first - stamp_s) > kRecordedMaxGapS) return std::nullopt;
    return best->second;
  }

  // @brief: center-crop the height to width/2 and resize to 512x256 (BGR),
  // the display counterpart of the window's preprocessing, for the overlay
  static cv::Mat CropResize(const cv::Mat &frame) {
    cv::Mat img = frame;
    const int target_h = img.cols / 2;
    if (img.rows > target_h) img = img(cv::Rect(0, (img.rows - target_h) / 2, img.cols, target_h));
    cv::Mat out;
    cv::resize(img, out, cv::Size(model::planner::kInputWidth, model::planner::kInputHeight),
               0, 0, cv::INTER_LINEAR);
    return out;
  }

  // @brief: every camera. The JPEG decode and the preprocessing run here, on
  // the camera's own thread and outside the lock; the lock is held only to
  // push the frame and
  // to stage a plan, never across an inference. A view-0 frame first settles
  // the plan still waiting for its side views (a newer front frame
  // supersedes it), then joins the window and -- if the pacing says so --
  // becomes the pending plan; a side frame joins the window and may complete
  // the pending plan
  void OnImage(int view, sensor_msgs::msg::CompressedImage::ConstSharedPtr msg) {
    const cv::Mat frame = cv::imdecode(cv::Mat(msg->data), cv::IMREAD_COLOR);
    if (frame.empty()) return;
    std::vector<float> chw(model::planner::kFrameNumel);
    if (!preprocess::PreprocessFrame(frame, chw.data())) return;
    const double stamp_s = rclcpp::Time(msg->header.stamp).seconds();

    std::lock_guard<std::mutex> lock(plan_mutex_);
    if (view != 0) {
      window_->PushPreprocessed(view, std::move(chw), stamp_s);
      Stage(false);
      return;
    }

    if (pending_) Stage(true);
    window_->PushPreprocessed(0, std::move(chw), stamp_s);
    if (last_input_s_ && stamp_s > *last_input_s_) input_gap_s_ = stamp_s - *last_input_s_;
    last_input_s_ = stamp_s;
    if (!TriggerDue(stamp_s, last_tick_s_, input_gap_s_, period_s_)) return;
    last_tick_s_ = stamp_s;

    if (!odom_) {
      RCLCPP_INFO_THROTTLE(get_logger(), *get_clock(), 2000,
                           "warming up: view 0 has %d frames, odometry not received yet",
                           window_->Size(0));
      return;
    }
    // The ego is sampled now, with the front frame, so a plan that waits for
    // its side views (or the recorded route points) still describes this instant
    pending_ = Pending{stamp_s, msg->header.stamp, frame, {}, ToSample(*odom_),
                       odom_->pose.pose.position.z};
    Stage(false);
  }

  // @brief: decide the pending plan, with plan_mutex_ held. A view-0 history
  // that does not reach back 1.5 s yet is the warm-up. A side slot without a
  // frame within tolerance waits while it may still arrive (the newest slot,
  // while that camera has not passed the front stamp + tolerance yet); after
  // that, or when `final` (a newer front frame came), it skips the plan or,
  // with missing_view_fill:=zeros, takes the blank frame there. In recorded
  // mode the plan also waits for the vehicle's route points of this tick,
  // which it published a few tens of ms after the frame. A plan that is
  // ready is assembled into a staging buffer and handed to the planning thread
  void Stage(bool final) {
    if (!pending_) return;
    const preprocess::ViewSelection sel = window_->Select();
    const auto may_still_arrive = [&](int view, int slot) {
      if (final || slot != model::planner::kNumHistFrames - 1) return false;
      const double newest = window_->NewestStamp(view);
      return std::isnan(newest) || newest < pending_->stamp_s + view_tolerance_s_;
    };
    if (!sel.ok) {
      if (sel.failed_view == 0) {
        RCLCPP_INFO_THROTTLE(get_logger(), *get_clock(), 2000, "warming up: %s", sel.reason.c_str());
        pending_.reset();
        return;
      }
      if (may_still_arrive(sel.failed_view, sel.failed_slot)) return;
      Skip(sel.reason);
      return;
    }
    for (const auto &[view, slot] : sel.filled) {
      if (may_still_arrive(view, slot)) return;
    }

    // --- status ---
    status::StatusResult st;
    if (route_source_ == "recorded") {
      const bool have_this_tick = !recorded_.empty() && recorded_.back().first >= pending_->stamp_s - 0.02;
      if (!have_this_tick && !final) return;
      const auto points = RecordedAt(pending_->stamp_s);
      if (!points) {
        Skip("no recorded route points within " + std::to_string(static_cast<int>(kRecordedMaxGapS * 1e3)) +
             " ms of the frame -- is the bag playing the vehicle planner's route_points topic? "
             "(route_source:=file takes a route.npy instead)");
        return;
      }
      st = status_->BuildWithRoutePoints(pending_->stamp_s, pending_->ego, *points);
    } else {
      st = status_->Build(pending_->stamp_s, pending_->ego);
    }
    if (!st.ok) {
      Skip("no status: " + st.error);
      return;
    }

    // --- hand over: the window into the spare buffer, the job to the planner
    // thread. If that thread is still busy with the previous plan, this one
    // replaces the job it has not started yet ---
    spare_video_.resize(planner_->VideoNumel());
    window_->Assemble(spare_video_.data());
    if (sel.worst_view_offset_s > worst_view_offset_s_) worst_view_offset_s_ = sel.worst_view_offset_s;
    filled_ += sel.filled.size();
    pending_->status = std::move(st);
    if (ready_) ++dropped_;
    ready_ = ReadyJob{std::move(*pending_), std::move(spare_video_), sel.filled.size()};
    spare_video_ = {};
    pending_.reset();
    plan_cv_.notify_one();
  }

  // @brief: the planning thread. It owns the engine's pinned input buffer:
  // the staged window is copied in and the inference runs without holding
  // plan_mutex_, so camera frames keep flowing into the window meanwhile
  void PlanLoop() {
    while (true) {
      ReadyJob job;
      {
        std::unique_lock<std::mutex> lock(plan_mutex_);
        plan_cv_.wait(lock, [this] { return plan_stop_ || ready_.has_value(); });
        if (plan_stop_) return;
        job = std::move(*ready_);
        ready_.reset();
      }
      std::memcpy(planner_->VideoInput(), job.video.data(), job.video.size() * sizeof(float));
      const auto t0 = std::chrono::steady_clock::now();
      const common::Trajectory plan = planner_->ForwardPrepared(job.pending.status.ego.Encode());
      const double infer_ms =
          std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();

      std::size_t planned = 0, skipped = 0, dropped = 0, filled = 0;
      double worst_view_offset_s = 0.0;
      {
        std::lock_guard<std::mutex> lock(plan_mutex_);
        if (spare_video_.empty()) spare_video_ = std::move(job.video);  // reuse the allocation
        if (plan.empty()) {
          Skip("the engine refused the status");
          continue;
        }
        planned = ++planned_;
        skipped = skipped_;
        dropped = dropped_;
        filled = filled_;
        worst_view_offset_s = worst_view_offset_s_;
      }
      const common::Trajectory &shown = Smoothed(plan, job.pending.stamp);
      PublishPlan(plan, job.pending);

      // Drawing and encoding cost more than the inference does; the viz thread
      // does them. The mailbox holds one job: if drawing falls behind, the
      // pending frame is replaced rather than queued, so the overlay stays live
      // and the planning loop never waits. `frame` is a cv::Mat, so this shares
      // rather than copies
      {
        std::lock_guard<std::mutex> lock(viz_mutex_);
        viz_job_ = VizJob{job.pending.frame, shown, job.pending.status.ego, job.filled};
      }
      viz_cv_.notify_one();

      RCLCPP_INFO_THROTTLE(get_logger(), *get_clock(), 2000,
                           "planning: v=%.1f m/s  |4s|=%.1f m  engine %.1f ms  (%d view%s, side views "
                           "within %.0f ms; %zu planned, %zu skipped, %zu superseded, %zu blank slots)",
                           std::hypot(job.pending.status.ego.vel_x, job.pending.status.ego.vel_y),
                           std::hypot(plan.back().x, plan.back().y), infer_ms, num_views_,
                           num_views_ == 1 ? "" : "s", worst_view_offset_s * 1e3, planned, skipped, dropped,
                           filled);
    }
  }

  // @brief: give up on the pending plan and say so. Every skipped inference
  // is counted and reported
  void Skip(const std::string &reason) {
    ++skipped_;
    RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000,
                         "inference skipped (%zu so far, %zu planned): %s", skipped_, planned_,
                         reason.c_str());
    pending_.reset();
  }

  // @brief: the plan to draw. The smoother carries the previous one forward by
  // the gap between camera stamps, so it needs a gap it can trust: a bag that
  // loops or seeks hands us a stamp jump, and carrying a plan across one would
  // smear a stale trajectory over the overlay for several frames
  const common::Trajectory &Smoothed(const common::Trajectory &plan,
                                     const builtin_interfaces::msg::Time &stamp) {
    const rclcpp::Time now_stamp(stamp);
    const double dt = have_stamp_ ? (now_stamp - prev_stamp_).seconds() : 0.0;
    if (have_stamp_ && (dt <= 0.0 || dt > kMaxGapS)) {
      RCLCPP_WARN(get_logger(), "camera stamp jumped %.2f s -- resetting the plan filter", dt);
      smoother_->Reset();
    }
    prev_stamp_ = now_stamp;
    have_stamp_ = true;

    shown_ = smoother_->Update(plan, dt > 0.0 ? dt : kNominalPeriodS);
    return shown_;
  }

  // @brief: ~/plan in base_link (the raw engine output), ~/plan_map and
  // ~/route_points in the map frame through the ego pose the plan is anchored
  // on, so RViz can draw them without a TF tree
  void PublishPlan(const common::Trajectory &plan, const Pending &job) {
    nav_msgs::msg::Path path, path_map;
    path.header.stamp = job.stamp;
    path.header.frame_id = "base_link";
    path_map.header.stamp = job.stamp;
    path_map.header.frame_id = "map";
    const double c = std::cos(job.ego.yaw), s = std::sin(job.ego.yaw);
    for (const auto &wp : plan) {
      geometry_msgs::msg::PoseStamped ps;
      ps.header = path.header;
      ps.pose.position.x = wp.x;
      ps.pose.position.y = wp.y;
      ps.pose.orientation.z = std::sin(wp.heading / 2);
      ps.pose.orientation.w = std::cos(wp.heading / 2);
      path.poses.push_back(ps);

      geometry_msgs::msg::PoseStamped pm;
      pm.header = path_map.header;
      pm.pose.position.x = job.ego.x + c * wp.x - s * wp.y;
      pm.pose.position.y = job.ego.y + s * wp.x + c * wp.y;
      pm.pose.position.z = job.ego_z;
      pm.pose.orientation.z = std::sin((job.ego.yaw + wp.heading) / 2);
      pm.pose.orientation.w = std::cos((job.ego.yaw + wp.heading) / 2);
      path_map.poses.push_back(pm);
    }
    pub_plan_->publish(path);
    pub_plan_map_->publish(path_map);

    geometry_msgs::msg::PoseArray points;
    points.header = path_map.header;
    for (const auto &p : job.status.route_points_map) {
      geometry_msgs::msg::Pose pose;
      pose.position.x = p.x;
      pose.position.y = p.y;
      pose.position.z = job.ego_z;
      pose.orientation.w = 1.0;
      points.poses.push_back(pose);
    }
    pub_route_points_->publish(points);
  }

  // @brief: the overlay projector from the view-0 camera_info and /tf_static,
  // built once both are there. Called from the viz thread
  void EnsureProjector() {
    if (projector_ || !tf_buffer_) return;
    sensor_msgs::msg::CameraInfo::ConstSharedPtr info;
    {
      std::lock_guard<std::mutex> lock(viz_mutex_);
      info = camera_info_;
    }
    if (!info || info->width == 0) return;
    geometry_msgs::msg::TransformStamped tf;
    try {
      tf = tf_buffer_->lookupTransform("base_link", info->header.frame_id, tf2::TimePointZero);
    } catch (const std::exception &e) {
      RCLCPP_INFO_THROTTLE(get_logger(), *get_clock(), 5000,
                           "camera overlay waits for base_link -> %s: %s",
                           info->header.frame_id.c_str(), e.what());
      return;
    }
    // K for the 512x256 crop: rows cropped to a centred width/2 band, then
    // the whole image scaled by 512 / width
    const double scale = static_cast<double>(model::planner::kInputWidth) / info->width;
    const double top = std::max(0.0, (info->height - info->width / 2.0) / 2.0);
    visualizer::CameraParams p;
    p.k = cv::Matx33d(info->k[0] * scale, info->k[1] * scale, info->k[2] * scale,
                      info->k[3] * scale, info->k[4] * scale, (info->k[5] - top) * scale,
                      0, 0, 1);
    const auto &q = tf.transform.rotation;
    p.r = cv::Matx33d(1 - 2 * (q.y * q.y + q.z * q.z), 2 * (q.x * q.y - q.z * q.w), 2 * (q.x * q.z + q.y * q.w),
                      2 * (q.x * q.y + q.z * q.w), 1 - 2 * (q.x * q.x + q.z * q.z), 2 * (q.y * q.z - q.x * q.w),
                      2 * (q.x * q.z - q.y * q.w), 2 * (q.y * q.z + q.x * q.w), 1 - 2 * (q.x * q.x + q.y * q.y));
    p.t = cv::Vec3d(tf.transform.translation.x, tf.transform.translation.y, tf.transform.translation.z);
    projector_ = visualizer::CreateCameraProjector(p);
    RCLCPP_INFO(get_logger(), "camera overlay from %s camera_info + tf (fx %.0f px on the full image)",
                info->header.frame_id.c_str(), info->k[0]);
  }

  // @brief: the viz thread. Sleeps until the inference leaves a job, draws
  // it, and goes back to sleep. `bev_` and `projector_` are used by this
  // thread alone
  void VizLoop() {
    while (true) {
      VizJob job;
      {
        std::unique_lock<std::mutex> lock(viz_mutex_);
        viz_cv_.wait(lock, [this] { return viz_stop_ || viz_job_.has_value(); });
        if (viz_stop_) return;
        job = std::move(*viz_job_);
        viz_job_.reset();
      }
      EnsureProjector();
      PublishViz(CropResize(job.frame), job);
    }
  }

  void PublishViz(cv::Mat camera, const VizJob &job) {
    const common::Trajectory &plan = job.plan;
    const common::EgoStatus &ego = job.ego;
    if (projector_) projector_->DrawRibbon(camera, plan, 0.9, 0.5);

    bev_->Clear();
    bev_->DrawTrajectoryHorizon(plan, 2);
    for (const common::RoutePoint &p : ego.route_points) bev_->DrawSubgoal(p.x_m, p.y_m);
    bev_->DrawVelocity(ego.vel_x, ego.vel_y);
    bev_->AddLegend("plan now->+4s", visualizer::HorizonColor(0.0));
    bev_->AddLegend(ego.route_points.size() == 1 ? "subgoal 50m" : "route points",
                    visualizer::color::kSubgoal);

    const int H = 560;
    cv::Mat cam_panel, bev_panel;
    cv::resize(camera, cam_panel, cv::Size(H * camera.cols / camera.rows, H));
    const cv::Mat &bev = bev_->Image();
    cv::resize(bev, bev_panel, cv::Size(H * bev.cols / bev.rows, H), 0, 0, cv::INTER_AREA);

    const float speed = std::hypot(ego.vel_x, ego.vel_y);
    const float wp = std::hypot(plan.back().x, plan.back().y);
    // km/h on the overlay, to match the offline visualizer; the log line stays
    // in m/s, which is the unit the model and the odometry work in
    cv::putText(cam_panel,
                cv::format("LatentDrive TRT %dv  v %.1f km/h  |4s| %.1f m%s", num_views_, speed * 3.6f, wp,
                           job.filled ? "  (blank side view)" : ""),
                {12, 28}, cv::FONT_HERSHEY_DUPLEX, 0.6, {230, 230, 230}, 1, cv::LINE_AA);

    cv::Mat frame(H, cam_panel.cols + 2 + bev_panel.cols, CV_8UC3, cv::Scalar(45, 47, 47));
    cam_panel.copyTo(frame(cv::Rect(0, 0, cam_panel.cols, H)));
    bev_panel.copyTo(frame(cv::Rect(cam_panel.cols + 2, 0, bev_panel.cols, H)));

    std_msgs::msg::Header header;
    header.stamp = now();
    header.frame_id = "base_link";

    // JPEG by default: the raw panel is 2.4 MB, which DDS delivers at about
    // 4 Hz on this machine, and the encoded one is nearer 150 kB
    sensor_msgs::msg::CompressedImage jpeg;
    jpeg.header = header;
    jpeg.format = "jpeg";
    cv::imencode(".jpg", frame, jpeg.data, {cv::IMWRITE_JPEG_QUALITY, viz_jpeg_quality_});
    pub_viz_jpeg_->publish(jpeg);

    // The raw topic is kept for anything that cannot read the compressed one,
    // but only paid for while something is actually subscribed to it
    if (pub_viz_->get_subscription_count() > 0) {
      sensor_msgs::msg::Image out;
      out.header = header;
      out.height = frame.rows;
      out.width = frame.cols;
      out.encoding = "bgr8";
      out.step = frame.cols * 3;
      out.data.assign(frame.data, frame.data + frame.total() * 3);
      pub_viz_->publish(out);
    }
  }

  std::shared_ptr<model::planner::Planner> planner_;
  std::shared_ptr<preprocess::ViewWindow> window_;
  std::shared_ptr<status::StatusBuilder> status_;
  int num_views_ = 1;
  std::vector<std::string> camera_topics_;
  double view_tolerance_s_ = model::planner::kDefaultViewToleranceS;
  std::string route_source_;

  // Camera-trigger pacing on the view-0 header stamps
  double period_s_ = 0.1;
  double input_gap_s_ = 0.1;
  std::optional<double> last_input_s_;
  std::optional<double> last_tick_s_;

  std::optional<Pending> pending_;
  std::optional<ReadyJob> ready_;
  std::vector<float> spare_video_;
  // window_, pending_, ready_, spare_video_, status_, odom_, accel_, recorded_, the tallies
  std::mutex plan_mutex_;
  std::condition_variable plan_cv_;
  std::thread plan_thread_;
  bool plan_stop_ = false;
  std::size_t planned_ = 0;
  std::size_t skipped_ = 0;
  std::size_t dropped_ = 0;
  std::size_t filled_ = 0;
  double worst_view_offset_s_ = 0.0;
  std::deque<std::pair<double, std::vector<status::Point2d>>> recorded_;
  static constexpr double kRecordedMaxGapS = 0.3;

  std::shared_ptr<visualizer::BevVisualizer> bev_;
  std::shared_ptr<visualizer::CameraProjector> projector_;
  std::shared_ptr<visualizer::TrajectorySmoother> smoother_;
  std::shared_ptr<tf2_ros::Buffer> tf_buffer_;
  std::shared_ptr<tf2_ros::TransformListener> tf_listener_;
  sensor_msgs::msg::CameraInfo::ConstSharedPtr camera_info_;

  // Camera runs at 10 Hz; the nominal period only stands in for the first
  // frame and for a gap the filter has just been reset over
  static constexpr double kNominalPeriodS = 0.1;
  static constexpr double kMaxGapS = 1.0;
  // Explicitly ROS time: subtracting two rclcpp::Time of different clock
  // types throws, and the default construction is system time
  rclcpp::Time prev_stamp_{0, 0, RCL_ROS_TIME};
  bool have_stamp_ = false;
  common::Trajectory shown_;

  nav_msgs::msg::Odometry::ConstSharedPtr odom_;
  geometry_msgs::msg::AccelWithCovarianceStamped::ConstSharedPtr accel_;

  std::vector<std::shared_ptr<rclcpp::executors::SingleThreadedExecutor>> camera_executors_;
  std::vector<std::thread> camera_threads_;
  std::vector<rclcpp::Subscription<sensor_msgs::msg::CompressedImage>::SharedPtr> sub_images_;
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr sub_odom_;
  rclcpp::Subscription<geometry_msgs::msg::AccelWithCovarianceStamped>::SharedPtr sub_accel_;
  rclcpp::Subscription<geometry_msgs::msg::PoseArray>::SharedPtr sub_recorded_;
  rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr sub_info_;
  std::optional<VizJob> viz_job_;
  std::mutex viz_mutex_;  // viz_job_, viz_stop_, camera_info_
  std::condition_variable viz_cv_;
  std::thread viz_thread_;
  bool viz_stop_ = false;
  int viz_jpeg_quality_ = 80;

  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr pub_viz_;
  rclcpp::Publisher<sensor_msgs::msg::CompressedImage>::SharedPtr pub_viz_jpeg_;
  rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr pub_plan_;
  rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr pub_plan_map_;
  rclcpp::Publisher<geometry_msgs::msg::PoseArray>::SharedPtr pub_route_points_;
};

int main(int argc, char **argv) {
  rclcpp::init(argc, argv);
  // Every camera already preprocesses on its own thread; OpenCV fanning each
  // small resize out over every core on top of that only adds contention
  cv::setNumThreads(1);
  // The cameras spin on their own executors (see the constructor); this one
  // takes odometry, acceleration, the route points and camera_info
  rclcpp::executors::SingleThreadedExecutor executor;
  auto node = std::make_shared<LatentDriveNode>();
  executor.add_node(node);
  executor.spin();
  node.reset();
  rclcpp::shutdown();
  return 0;
}
