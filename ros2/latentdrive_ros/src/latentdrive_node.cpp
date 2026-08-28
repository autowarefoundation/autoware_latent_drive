// Minimal ROS 2 node for open-loop rosbag replay of the LatentDrive planner.
//
// Subscribes to the front camera, ego odometry and acceleration; every camera
// frame it slides the 4-frame window, builds the status vector (subgoal from a
// preloaded ego route, velocity + acceleration from the messages), runs the
// engine, and publishes the plan plus a camera+BEV visualization image.
//
// The subgoal comes from route.npy (the ego's own driven path, exported by
// scripts/export_route.py) because the container has no autoware_planning_msgs
// to parse /planning/trajectory. For open-loop replay the whole run is known,
// so "50 m of arc length ahead on the driven path" is a faithful route target.
#include <cmath>
#include <condition_variable>
#include <deque>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <vector>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <nlohmann/json.hpp>
#include <rclcpp/rclcpp.hpp>

#include <geometry_msgs/msg/accel_with_covariance_stamped.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <nav_msgs/msg/path.hpp>
#include <sensor_msgs/msg/compressed_image.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <std_msgs/msg/header.hpp>

#include "bev_visualizer.hpp"
#include "camera_projector.hpp"
#include "color_map.hpp"
#include "frame_window.hpp"
#include "npy.hpp"
#include "planner.hpp"
#include "trajectory.hpp"
#include "trajectory_smoother.hpp"

using namespace std::chrono_literals;

namespace {

// @brief: This function is used to load the prd_jt / CAM_FRONT_WIDE entry from
// the deployment calibration.json (K_crop_resized for the 512x256 image and the
// camera-to-ego extrinsics), returning a projector for the camera overlay
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

}  // namespace

class LatentDriveNode : public rclcpp::Node {
 public:
  LatentDriveNode() : Node("latentdrive_node") {
    const auto engine = declare_parameter<std::string>("engine_file", "");
    const auto calib = declare_parameter<std::string>("calibration_file", "");
    const auto route = declare_parameter<std::string>("route_file", "");
    const auto scene_match = declare_parameter<std::string>("scene_match", "prd_jt");
    subgoal_ahead_m_ = declare_parameter<double>("subgoal_ahead_m", 50.0);
    const auto smooth_alpha = declare_parameter<double>("smooth_alpha", 0.35);
    viz_jpeg_quality_ = static_cast<int>(declare_parameter<int64_t>("viz_jpeg_quality", 80));
    const auto viz_reliable = declare_parameter<bool>("viz_reliable", false);
    const auto camera_topic = declare_parameter<std::string>(
        "camera_topic", "/sensing/camera/camera1/image_raw/compressed");
    const auto odom_topic = declare_parameter<std::string>(
        "odom_topic", "/localization/kinematic_state");
    const auto accel_topic = declare_parameter<std::string>(
        "accel_topic", "/localization/acceleration");

    planner_ = model::planner::CreatePlanner(engine);
    window_ = preprocess::CreateFrameWindow();
    bev_ = visualizer::CreateBevVisualizer();
    // Display only: ~/plan carries the raw engine output, the overlay is what
    // gets filtered. Consecutive plans disagree by about a metre on how far
    // they reach, which makes the BEV tail pulse in and out
    smoother_ = visualizer::CreateTrajectorySmoother({smooth_alpha});
    projector_ = LoadProjector(calib, scene_match);
    if (!planner_ || !window_ || !bev_) {
      throw std::runtime_error("failed to init planner/window/bev");
    }
    if (!projector_) {
      RCLCPP_WARN(get_logger(), "no calibration entry matching '%s' -- camera overlay off",
                  scene_match.c_str());
    }
    LoadRoute(route);

    // warmup so the first real inference is not slow
    const std::vector<float> zeros(model::planner::kVideoNumel, 0.f);
    for (int i = 0; i < 5; ++i) planner_->Forward(zeros.data(), {});

    rclcpp::QoS sensor_qos = rclcpp::SensorDataQoS();
    sub_img_ = create_subscription<sensor_msgs::msg::CompressedImage>(
        camera_topic, sensor_qos,
        std::bind(&LatentDriveNode::OnImage, this, std::placeholders::_1));
    sub_odom_ = create_subscription<nav_msgs::msg::Odometry>(
        odom_topic, sensor_qos,
        [this](nav_msgs::msg::Odometry::ConstSharedPtr m) { odom_ = m; });
    sub_accel_ = create_subscription<geometry_msgs::msg::AccelWithCovarianceStamped>(
        accel_topic, sensor_qos,
        [this](geometry_msgs::msg::AccelWithCovarianceStamped::ConstSharedPtr m) { accel_ = m; });

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

    viz_thread_ = std::thread(&LatentDriveNode::VizLoop, this);
    RCLCPP_INFO(get_logger(), "latentdrive_node ready (%d waypoints)", planner_->NumWaypoints());
  }

  ~LatentDriveNode() override {
    {
      std::lock_guard<std::mutex> lock(viz_mutex_);
      viz_stop_ = true;
    }
    viz_cv_.notify_one();
    if (viz_thread_.joinable()) viz_thread_.join();
  }

 private:
  void LoadRoute(const std::string &path) {
    if (path.empty()) {
      RCLCPP_WARN(get_logger(), "no route_file -- subgoal will be straight ahead");
      return;
    }
    const auto a = utils::LoadNpy(path);  // [N,3] map_x,map_y,yaw
    route_.resize(a.shape[0]);
    for (std::size_t i = 0; i < a.shape[0]; ++i) {
      route_[i] = {a.data[i * 3 + 0], a.data[i * 3 + 1]};
    }
    RCLCPP_INFO(get_logger(), "loaded route: %zu points", route_.size());
  }

  // @brief: subgoal in the ego frame = the point `subgoal_ahead_m_` of arc
  // length ahead of the nearest route point, rotated into the current ego frame
  std::pair<float, float> Subgoal(double ex, double ey, double eyaw) const {
    if (route_.size() < 2) return {static_cast<float>(subgoal_ahead_m_), 0.f};
    std::size_t nearest = 0;
    double best = 1e18;
    for (std::size_t i = 0; i < route_.size(); ++i) {
      const double d = std::hypot(route_[i].first - ex, route_[i].second - ey);
      if (d < best) { best = d; nearest = i; }
    }
    double acc = 0.0;
    std::size_t j = nearest;
    for (; j + 1 < route_.size() && acc < subgoal_ahead_m_; ++j) {
      acc += std::hypot(route_[j + 1].first - route_[j].first,
                        route_[j + 1].second - route_[j].second);
    }
    const double dx = route_[j].first - ex, dy = route_[j].second - ey;
    const double c = std::cos(eyaw), s = std::sin(eyaw);
    return {static_cast<float>(c * dx + s * dy),       // forward
            static_cast<float>(-s * dx + c * dy)};     // left
  }

  // @brief: center-crop the height to width/2 and resize to 512x256 (BGR),
  // the display counterpart of FrameWindow's preprocessing, for the overlay
  static cv::Mat CropResize(const cv::Mat &frame) {
    cv::Mat img = frame;
    const int target_h = img.cols / 2;
    if (img.rows > target_h) img = img(cv::Rect(0, (img.rows - target_h) / 2, img.cols, target_h));
    cv::Mat out;
    cv::resize(img, out, cv::Size(model::planner::kInputWidth, model::planner::kInputHeight),
               0, 0, cv::INTER_LINEAR);
    return out;
  }

  void OnImage(sensor_msgs::msg::CompressedImage::ConstSharedPtr msg) {
    const cv::Mat frame = cv::imdecode(cv::Mat(msg->data), cv::IMREAD_COLOR);
    if (frame.empty()) return;
    window_->Push(frame);
    if (!window_->Ready() || !odom_) {
      RCLCPP_INFO_THROTTLE(get_logger(), *get_clock(), 2000,
                           "warming up: window %d/16, odom=%s", window_->Size(),
                           odom_ ? "ready" : "waiting");
      return;
    }

    // --- status ---
    const auto &q = odom_->pose.pose.orientation;
    const double eyaw = std::atan2(2 * (q.w * q.z + q.x * q.y),
                                   1 - 2 * (q.y * q.y + q.z * q.z));
    const auto sg = Subgoal(odom_->pose.pose.position.x, odom_->pose.pose.position.y, eyaw);
    common::EgoStatus ego;
    ego.subgoal_x_m = sg.first;
    ego.subgoal_y_m = sg.second;
    ego.vel_x = static_cast<float>(odom_->twist.twist.linear.x);
    ego.vel_y = static_cast<float>(odom_->twist.twist.linear.y);
    if (accel_) {
      ego.acc_x = static_cast<float>(accel_->accel.accel.linear.x);
      ego.acc_y = static_cast<float>(accel_->accel.accel.linear.y);
    }

    // --- inference ---
    window_->Assemble(planner_->VideoInput());
    const common::Trajectory plan = planner_->ForwardPrepared(ego.Encode());

    const common::Trajectory &shown = Smoothed(plan, msg->header.stamp);
    PublishPlan(plan, msg->header.stamp);

    // Drawing and encoding cost more than the inference does, and doing them
    // here used to block this callback long enough to drop camera frames --
    // which is not just a stutter, it puts the wrong spacing into the 4-frame
    // history the model reads. Hand the work to the viz thread instead. The
    // mailbox holds one job: if drawing falls behind, the pending frame is
    // replaced rather than queued, so the overlay stays live and the inference
    // loop never waits. `frame` is a cv::Mat, so this shares rather than copies
    {
      std::lock_guard<std::mutex> lock(viz_mutex_);
      viz_job_ = VizJob{frame, shown, ego};
    }
    viz_cv_.notify_one();

    RCLCPP_INFO_THROTTLE(get_logger(), *get_clock(), 2000,
                         "planning: v=%.1f m/s  |4s|=%.1f m", std::hypot(ego.vel_x, ego.vel_y),
                         std::hypot(plan.back().x, plan.back().y));
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

  void PublishPlan(const common::Trajectory &plan, const rclcpp::Time &stamp) {
    nav_msgs::msg::Path path;
    path.header.stamp = stamp;
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

  // @brief: the viz thread. Sleeps until the image callback leaves a job, draws
  // it, and goes back to sleep. Nothing here is touched by the inference
  // callback -- `bev_` is drawn on by this thread alone, `projector_` is
  // read-only after construction
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
      PublishViz(CropResize(job.frame), job.plan, job.ego);
    }
  }

  void PublishViz(cv::Mat camera, const common::Trajectory &plan, const common::EgoStatus &ego) {
    if (projector_) projector_->DrawRibbon(camera, plan, 0.9, 0.5);

    bev_->Clear();
    bev_->DrawTrajectoryHorizon(plan, 2);
    bev_->DrawSubgoal(ego.subgoal_x_m, ego.subgoal_y_m);
    bev_->DrawVelocity(ego.vel_x, ego.vel_y);
    bev_->AddLegend("plan now->+4s", visualizer::HorizonColor(0.0));
    bev_->AddLegend("subgoal 50m", visualizer::color::kSubgoal);

    const int H = 560;
    cv::Mat cam_panel, bev_panel;
    cv::resize(camera, cam_panel, cv::Size(H * camera.cols / camera.rows, H));
    const cv::Mat &bev = bev_->Image();
    cv::resize(bev, bev_panel, cv::Size(H * bev.cols / bev.rows, H), 0, 0, cv::INTER_AREA);

    const float speed = std::hypot(ego.vel_x, ego.vel_y);
    const float wp = std::hypot(plan.back().x, plan.back().y);
    // km/h on the overlay, to match the offline visualizer; the log line below
    // stays in m/s, which is the unit the model and the odometry work in
    cv::putText(cam_panel, cv::format("LatentDrive TRT  v %.1f km/h  |4s| %.1f m", speed * 3.6f, wp),
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
  std::shared_ptr<preprocess::FrameWindow> window_;
  std::shared_ptr<visualizer::BevVisualizer> bev_;
  std::shared_ptr<visualizer::CameraProjector> projector_;
  std::shared_ptr<visualizer::TrajectorySmoother> smoother_;
  std::vector<std::pair<double, double>> route_;
  double subgoal_ahead_m_ = 50.0;

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

  rclcpp::Subscription<sensor_msgs::msg::CompressedImage>::SharedPtr sub_img_;
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr sub_odom_;
  rclcpp::Subscription<geometry_msgs::msg::AccelWithCovarianceStamped>::SharedPtr sub_accel_;
  // One in-flight drawing job, latest wins
  struct VizJob {
    cv::Mat frame;  // the decoded camera frame, full resolution
    common::Trajectory plan;
    common::EgoStatus ego;
  };
  std::optional<VizJob> viz_job_;
  std::mutex viz_mutex_;
  std::condition_variable viz_cv_;
  std::thread viz_thread_;
  bool viz_stop_ = false;
  int viz_jpeg_quality_ = 80;

  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr pub_viz_;
  rclcpp::Publisher<sensor_msgs::msg::CompressedImage>::SharedPtr pub_viz_jpeg_;
  rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr pub_plan_;
};

int main(int argc, char **argv) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<LatentDriveNode>());
  rclcpp::shutdown();
  return 0;
}
