// Minimal ROS 2 node for open-loop rosbag replay of the LatentDrive planner.
//
// Subscribes to the cameras, the ego odometry and acceleration, and the route
// points the vehicle's planner fed its model (recorded in the bags of
// LatentDrive runs on /tensorrt_e2e/debug/latentdrive/route_points). Every
// view-0 (CAM_FRONT) frame triggers a plan: the timestamped window picks the
// four history slots, the status builder makes the status vector, the engine
// runs, and the node publishes the plan and draws it on the front camera
// image -- in an OpenCV window and/or an mp4 file.
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
// Route points: the vehicle stamped the ones it fed its model with its tick,
// a few tens of ms after the camera frame it planned on; a plan waits for the
// message of its tick. Motion terms (`motion_source`): pose_difference (the
// training convention, differenced positions 0.1 s apart) or twist (the
// odometry twist and the acceleration topic).
//
// Overlay: the plan projected on the ground plane of the view-0 image, with
// the intrinsics from the bag's camera_info and the camera pose from its
// /tf_static (base_link -> camera).
#include <algorithm>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstddef>
#include <cstring>
#include <deque>
#include <memory>
#include <mutex>
#include <optional>
#include <pthread.h>
#include <string>
#include <thread>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

#include <rclcpp/rclcpp.hpp>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>

#include <geometry_msgs/msg/accel_with_covariance_stamped.hpp>
#include <geometry_msgs/msg/pose_array.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <nav_msgs/msg/path.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <sensor_msgs/msg/compressed_image.hpp>

#include "camera_projector.hpp"
#include "frame_window.hpp"
#include "planner.hpp"
#include "status_builder.hpp"
#include "trajectory.hpp"
#include "trajectory_smoother.hpp"

using namespace std::chrono_literals;

namespace {

constexpr char kWindowName[] = "LatentDrive replay";

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

// @brief: center-crop the rows to a 2:1 band and resize to `width` x
// `width / 2` (BGR): the display counterpart of the model's preprocessing,
// so the overlay's intrinsics are the model crop's, scaled
cv::Mat CropResize(const cv::Mat &frame, int width) {
  cv::Mat img = frame;
  const int band = img.cols / 2;
  if (img.rows > band) img = img(cv::Rect(0, (img.rows - band) / 2, img.cols, band));
  cv::Mat out;
  cv::resize(img, out, cv::Size(width, width / 2), 0, 0, cv::INTER_LINEAR);
  return out;
}

}  // namespace

class LatentDriveNode : public rclcpp::Node {
 public:
  LatentDriveNode() : Node("latentdrive_node") {
    const auto engine = declare_parameter<std::string>("engine_file", "");
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
    const auto odom_topic = declare_parameter<std::string>("odom_topic", "/localization/kinematic_state");
    const auto accel_topic = declare_parameter<std::string>("accel_topic", "/localization/acceleration");
    route_points_topic_ = declare_parameter<std::string>(
        "route_points_topic", "/tensorrt_e2e/debug/latentdrive/route_points");
    const auto motion_source = declare_parameter<std::string>("motion_source", "pose_difference");
    const auto smooth_alpha = declare_parameter<double>("smooth_alpha", 0.35);
    show_window_ = declare_parameter<bool>("show_window", true);
    out_video_ = declare_parameter<std::string>("out_video", "");
    viz_width_ = static_cast<int>(declare_parameter<int64_t>("viz_width", 1024));

    if (missing_view_fill != "skip" && missing_view_fill != "zeros") {
      throw std::runtime_error("missing_view_fill must be skip or zeros, got " + missing_view_fill);
    }
    if (motion_source != "pose_difference" && motion_source != "twist") {
      throw std::runtime_error("motion_source must be pose_difference or twist, got " + motion_source);
    }
    period_s_ = max_rate_hz > 0.0 ? 1.0 / max_rate_hz : 0.0;
    video_fps_ = max_rate_hz > 0.0 ? max_rate_hz : 10.0;

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
    status_params.motion_source = motion_source == "twist" ? status::MotionSource::kTwist
                                                           : status::MotionSource::kPoseDifference;
    status_ = status::CreateStatusBuilder(status_params);

    // Display only: ~/plan carries the raw engine output, the overlay is what
    // gets filtered. Consecutive plans disagree by about a metre on how far
    // they reach, which makes the drawn tail pulse in and out
    smoother_ = visualizer::CreateTrajectorySmoother({smooth_alpha});
    if (!window_ || !status_ || !smoother_) {
      throw std::runtime_error("failed to init the window / status builder / smoother");
    }

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
    sub_route_points_ = create_subscription<geometry_msgs::msg::PoseArray>(
        route_points_topic_, rclcpp::QoS(10),
        [this](geometry_msgs::msg::PoseArray::ConstSharedPtr m) { OnRoutePoints(m); });

    // The overlay comes from the bag itself: the view-0 camera_info
    // (intrinsics) and /tf_static (base_link -> camera)
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

    pub_plan_ = create_publisher<nav_msgs::msg::Path>("~/plan", 1);

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
                "at most %.1f Hz on the view-0 stamps; overlay %s%s%s",
                num_views_, num_views_ == 1 ? "" : "s", views.c_str(), view_tolerance_s_ * 1e3,
                missing_view_fill.c_str(), planner_->StatusDim(), planner_->NumRoutePoints(),
                planner_->NumRoutePoints() == 1 ? "" : "s", route_points_topic_.c_str(),
                motion_source.c_str(), planner_->NumWaypoints(), max_rate_hz,
                show_window_ ? "in a window" : "", show_window_ && !out_video_.empty() ? " and " : "",
                out_video_.empty() ? (show_window_ ? "" : "off") : ("-> " + out_video_).c_str());
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
  // stamp; the status is filled in when the plan is staged
  struct Pending {
    double stamp_s;
    builtin_interfaces::msg::Time stamp;
    cv::Mat frame;
    status::StatusResult status;
    status::PoseSample ego;
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

  void OnOdometry(nav_msgs::msg::Odometry::ConstSharedPtr m) {
    std::lock_guard<std::mutex> lock(plan_mutex_);
    odom_ = m;
    if (!odom_history_.empty() && Seconds(m->header.stamp) < Seconds(odom_history_.back()->header.stamp) - 1.0) {
      odom_history_.clear();  // the bag looped or jumped back
    }
    odom_history_.push_back(m);
    while (odom_history_.size() > 100) odom_history_.pop_front();  // 2 s at 50 Hz
    status_->PushPose(ToSample(*m));
  }

  static double Seconds(const builtin_interfaces::msg::Time &t) { return t.sec + t.nanosec * 1e-9; }

  // @brief: the odometry stamped nearest to `stamp_s`. On the vehicle the
  // planner gets a camera frame a few ms after its stamp and takes the newest
  // odometry then, a few ms past the frame (measured on a LatentDrive run:
  // +4 ms). In a bag the frames were logged ~100 ms after their stamps --
  // the recorder's own latency -- and are played that late, so the newest
  // odometry at arrival is ~0.1 s too new: 1-2 m along the route at speed
  nav_msgs::msg::Odometry::ConstSharedPtr OdomAt(double stamp_s) const {
    nav_msgs::msg::Odometry::ConstSharedPtr best = odom_;
    for (const auto &o : odom_history_) {
      if (std::abs(Seconds(o->header.stamp) - stamp_s) < std::abs(Seconds(best->header.stamp) - stamp_s)) best = o;
    }
    return best;
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

  void OnRoutePoints(geometry_msgs::msg::PoseArray::ConstSharedPtr m) {
    std::lock_guard<std::mutex> lock(plan_mutex_);
    const double stamp_s = rclcpp::Time(m->header.stamp).seconds();
    if (!route_points_.empty() && stamp_s < route_points_.back().first - 1.0) route_points_.clear();
    std::vector<status::Point2d> points;
    for (const auto &p : m->poses) points.push_back({p.position.x, p.position.y});
    route_points_.emplace_back(stamp_s, std::move(points));
    while (route_points_.size() > 50) route_points_.pop_front();
  }

  // @brief: the recorded route points nearest in time to `stamp_s`. The
  // vehicle stamped them with its tick, a few ms after the camera frame it
  // planned on; more than kRoutePointsMaxGapS away and they belong to another tick
  std::optional<std::vector<status::Point2d>> RoutePointsAt(double stamp_s) const {
    const std::pair<double, std::vector<status::Point2d>> *best = nullptr;
    for (const auto &r : route_points_) {
      if (!best || std::abs(r.first - stamp_s) < std::abs(best->first - stamp_s)) best = &r;
    }
    if (!best || std::abs(best->first - stamp_s) > kRoutePointsMaxGapS) return std::nullopt;
    return best->second;
  }

  // @brief: every camera. The JPEG decode and the preprocessing run here, on
  // the camera's own thread and outside the lock; the lock is held only to
  // push the frame and to stage a plan, never across an inference. A view-0
  // frame first settles the plan still waiting for its side views (a newer
  // front frame supersedes it), then joins the window and -- if the pacing
  // says so -- becomes the pending plan; a side frame joins the window and
  // may complete the pending plan
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
    // The ego is sampled at the front frame's stamp, so a plan that waits
    // for its side views (or the route points) still describes this instant
    const auto odom = OdomAt(stamp_s);
    pending_ = Pending{stamp_s, msg->header.stamp, frame, {}, ToSample(*odom)};
    Stage(false);
  }

  // @brief: decide the pending plan, with plan_mutex_ held. A view-0 history
  // that does not reach back 1.5 s yet is the warm-up. A side slot without a
  // frame within tolerance waits while it may still arrive (the newest slot,
  // while that camera has not passed the front stamp + tolerance yet); after
  // that, or when `final` (a newer front frame came), it skips the plan or,
  // with missing_view_fill:=zeros, takes the blank frame there. The plan also
  // waits for the vehicle's route points of this tick, which it published a
  // few tens of ms after the frame. A plan that is ready is assembled into a
  // staging buffer and handed to the planning thread
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
    const bool have_this_tick = !route_points_.empty() && route_points_.back().first >= pending_->stamp_s - 0.02;
    if (!have_this_tick && !final) return;
    const auto points = RoutePointsAt(pending_->stamp_s);
    if (!points) {
      Skip("no route points within " + std::to_string(static_cast<int>(kRoutePointsMaxGapS * 1e3)) +
           " ms of the frame -- is the bag playing " + route_points_topic_ + "?");
      return;
    }
    status::StatusResult st = status_->BuildWithRoutePoints(pending_->stamp_s, pending_->ego, *points);
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

      // Drawing costs more than the inference does; the viz thread does it.
      // The mailbox holds one job: if drawing falls behind, the pending frame
      // is replaced rather than queued, so the overlay stays live and the
      // planning loop never waits. `frame` is a cv::Mat, so this shares
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

  // @brief: ~/plan in base_link: the raw engine output, stamped with the
  // frame it was planned on
  void PublishPlan(const common::Trajectory &plan, const Pending &job) {
    nav_msgs::msg::Path path;
    path.header.stamp = job.stamp;
    path.header.frame_id = "base_link";
    for (const auto &wp : plan) {
      geometry_msgs::msg::PoseStamped ps;
      ps.header = path.header;
      ps.pose.position.x = wp.x;
      ps.pose.position.y = wp.y;
      ps.pose.orientation.z = std::sin(wp.heading / 2);
      ps.pose.orientation.w = std::cos(wp.heading / 2);
      path.poses.push_back(ps);
    }
    pub_plan_->publish(path);
  }

  // @brief: the overlay projector from the view-0 camera_info and /tf_static,
  // built once both are there. Called from the viz thread
  void EnsureProjector() {
    if (projector_) return;
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
    // K for the drawn image: rows cropped to a centred width/2 band, then the
    // whole image scaled to viz_width
    const double scale = static_cast<double>(viz_width_) / info->width;
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
  // it, and goes back to sleep; between jobs it keeps the window's event loop
  // alive. HighGUI is used by this thread alone
  void VizLoop() {
    while (true) {
      std::optional<VizJob> job;
      {
        std::unique_lock<std::mutex> lock(viz_mutex_);
        viz_cv_.wait_for(lock, 100ms, [this] { return viz_stop_ || viz_job_.has_value(); });
        if (viz_stop_) break;
        job.swap(viz_job_);
      }
      if (job) {
        EnsureProjector();
        Draw(*job);
      } else if (window_open_) {
        PollWindow();
      }
    }
    if (writer_.isOpened()) {
      writer_.release();
      RCLCPP_INFO(get_logger(), "wrote %s", out_video_.c_str());
    }
    if (window_open_) cv::destroyAllWindows();
  }

  void Draw(const VizJob &job) {
    cv::Mat image = CropResize(job.frame, viz_width_);
    if (projector_) projector_->DrawRibbon(image, job.plan, 0.9, 0.5);
    const float speed = std::hypot(job.ego.vel_x, job.ego.vel_y);
    const float reach = std::hypot(job.plan.back().x, job.plan.back().y);
    // km/h on the overlay, as a driver would read it; the log line stays in
    // m/s, the unit the model and the odometry work in
    cv::putText(image,
                cv::format("LatentDrive TRT %dv  v %.1f km/h  |4s| %.1f m%s", num_views_, speed * 3.6f, reach,
                           job.filled ? "  (blank side view)" : ""),
                {12, 28}, cv::FONT_HERSHEY_DUPLEX, 0.7, {230, 230, 230}, 1, cv::LINE_AA);

    if (!out_video_.empty() && !writer_failed_) {
      if (!writer_.isOpened()) {
        writer_.open(out_video_, cv::VideoWriter::fourcc('m', 'p', '4', 'v'), video_fps_, image.size());
        if (!writer_.isOpened()) {
          RCLCPP_ERROR(get_logger(), "cannot open %s for writing", out_video_.c_str());
          writer_failed_ = true;
        }
      }
      if (writer_.isOpened()) writer_.write(image);
    }
    if (show_window_) {
      try {
        cv::imshow(kWindowName, image);
        window_open_ = true;
        PollWindow();
      } catch (const cv::Exception &e) {
        RCLCPP_WARN(get_logger(), "no window (%s): drawing continues without one; show_window:=false "
                    "silences this", e.what());
        show_window_ = false;
      }
    }
  }

  // @brief: pump the window's events; Esc or q ends the node
  void PollWindow() {
    const int key = cv::waitKey(1);
    if (key == 27 || key == 'q') {
      RCLCPP_INFO(get_logger(), "window closed by key -- shutting down");
      rclcpp::shutdown();
    }
  }

  std::shared_ptr<model::planner::Planner> planner_;
  std::shared_ptr<preprocess::ViewWindow> window_;
  std::shared_ptr<status::StatusBuilder> status_;
  int num_views_ = 1;
  std::vector<std::string> camera_topics_;
  double view_tolerance_s_ = model::planner::kDefaultViewToleranceS;
  std::string route_points_topic_;

  // Camera-trigger pacing on the view-0 header stamps
  double period_s_ = 0.1;
  double input_gap_s_ = 0.1;
  std::optional<double> last_input_s_;
  std::optional<double> last_tick_s_;

  std::optional<Pending> pending_;
  std::optional<ReadyJob> ready_;
  std::vector<float> spare_video_;
  // window_, pending_, ready_, spare_video_, status_, odom_, odom_history_, accel_, route_points_, the tallies
  std::mutex plan_mutex_;
  std::condition_variable plan_cv_;
  std::thread plan_thread_;
  bool plan_stop_ = false;
  std::size_t planned_ = 0;
  std::size_t skipped_ = 0;
  std::size_t dropped_ = 0;
  std::size_t filled_ = 0;
  double worst_view_offset_s_ = 0.0;
  std::deque<std::pair<double, std::vector<status::Point2d>>> route_points_;
  static constexpr double kRoutePointsMaxGapS = 0.3;

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
  std::deque<nav_msgs::msg::Odometry::ConstSharedPtr> odom_history_;
  geometry_msgs::msg::AccelWithCovarianceStamped::ConstSharedPtr accel_;

  std::vector<std::shared_ptr<rclcpp::executors::SingleThreadedExecutor>> camera_executors_;
  std::vector<std::thread> camera_threads_;
  std::vector<rclcpp::Subscription<sensor_msgs::msg::CompressedImage>::SharedPtr> sub_images_;
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr sub_odom_;
  rclcpp::Subscription<geometry_msgs::msg::AccelWithCovarianceStamped>::SharedPtr sub_accel_;
  rclcpp::Subscription<geometry_msgs::msg::PoseArray>::SharedPtr sub_route_points_;
  rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr sub_info_;
  rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr pub_plan_;

  // The drawing: viz_job_, viz_stop_ and camera_info_ behind viz_mutex_; the
  // window, the writer and the projector belong to the viz thread
  std::optional<VizJob> viz_job_;
  std::mutex viz_mutex_;
  std::condition_variable viz_cv_;
  std::thread viz_thread_;
  bool viz_stop_ = false;
  bool show_window_ = true;
  bool window_open_ = false;
  std::string out_video_;
  double video_fps_ = 10.0;
  cv::VideoWriter writer_;
  bool writer_failed_ = false;
  int viz_width_ = 1024;
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
