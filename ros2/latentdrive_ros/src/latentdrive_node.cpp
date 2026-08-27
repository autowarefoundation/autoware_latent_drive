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
#include <deque>
#include <memory>
#include <string>
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

#include "bev_visualizer.hpp"
#include "camera_projector.hpp"
#include "color_map.hpp"
#include "frame_window.hpp"
#include "npy.hpp"
#include "planner.hpp"
#include "trajectory.hpp"

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
    const auto camera_topic = declare_parameter<std::string>(
        "camera_topic", "/sensing/camera/camera1/image_raw/compressed");
    const auto odom_topic = declare_parameter<std::string>(
        "odom_topic", "/localization/kinematic_state");
    const auto accel_topic = declare_parameter<std::string>(
        "accel_topic", "/localization/acceleration");

    planner_ = model::planner::CreatePlanner(engine);
    window_ = preprocess::CreateFrameWindow();
    bev_ = visualizer::CreateBevVisualizer();
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

    pub_viz_ = create_publisher<sensor_msgs::msg::Image>("~/viz", 1);
    pub_plan_ = create_publisher<nav_msgs::msg::Path>("~/plan", 1);
    RCLCPP_INFO(get_logger(), "latentdrive_node ready (%d waypoints)", planner_->NumWaypoints());
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

    PublishPlan(plan, msg->header.stamp);
    PublishViz(CropResize(frame), plan, ego);

    RCLCPP_INFO_THROTTLE(get_logger(), *get_clock(), 2000,
                         "planning: v=%.1f m/s  |4s|=%.1f m", std::hypot(ego.vel_x, ego.vel_y),
                         std::hypot(plan.back().x, plan.back().y));
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
    cv::putText(cam_panel, cv::format("LatentDrive TRT  v %.1f m/s  |4s| %.1f m", speed, wp),
                {12, 28}, cv::FONT_HERSHEY_DUPLEX, 0.6, {230, 230, 230}, 1, cv::LINE_AA);

    cv::Mat frame(H, cam_panel.cols + 2 + bev_panel.cols, CV_8UC3, cv::Scalar(45, 47, 47));
    cam_panel.copyTo(frame(cv::Rect(0, 0, cam_panel.cols, H)));
    bev_panel.copyTo(frame(cv::Rect(cam_panel.cols + 2, 0, bev_panel.cols, H)));

    auto out = std::make_shared<sensor_msgs::msg::Image>();
    out->header.stamp = now();
    out->header.frame_id = "base_link";
    out->height = frame.rows;
    out->width = frame.cols;
    out->encoding = "bgr8";
    out->step = frame.cols * 3;
    out->data.assign(frame.data, frame.data + frame.total() * 3);
    pub_viz_->publish(*out);
  }

  std::shared_ptr<model::planner::Planner> planner_;
  std::shared_ptr<preprocess::FrameWindow> window_;
  std::shared_ptr<visualizer::BevVisualizer> bev_;
  std::shared_ptr<visualizer::CameraProjector> projector_;
  std::vector<std::pair<double, double>> route_;
  double subgoal_ahead_m_ = 50.0;

  nav_msgs::msg::Odometry::ConstSharedPtr odom_;
  geometry_msgs::msg::AccelWithCovarianceStamped::ConstSharedPtr accel_;

  rclcpp::Subscription<sensor_msgs::msg::CompressedImage>::SharedPtr sub_img_;
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr sub_odom_;
  rclcpp::Subscription<geometry_msgs::msg::AccelWithCovarianceStamped>::SharedPtr sub_accel_;
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr pub_viz_;
  rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr pub_plan_;
};

int main(int argc, char **argv) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<LatentDriveNode>());
  rclcpp::shutdown();
  return 0;
}
