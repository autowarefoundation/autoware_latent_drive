// Model I/O contract of the LatentDrive e2e planner (e2e-idx8-hist4).
//
// The engine is exported with fixed shapes:
//   video  float32 [1, 3, 4, 256, 512]  4 front-camera frames, oldest..newest,
//                                       0.5 s apart, RGB, ImageNet-normalized
//   status float32 [1, 6]               [subgoal_x/10, subgoal_y/10,
//                                        vel_x, vel_y, acc_x, acc_y] (ego frame)
//   traj   float32 [1, 8, 3]            waypoints at t+0.5..4.0 s,
//                                       (x fwd [m], y left [m], heading [rad])
#pragma once

#include <array>
#include <cstddef>

namespace latentdrive {

inline constexpr int kInputHeight = 256;
inline constexpr int kInputWidth = 512;
inline constexpr int kNumHistFrames = 4;    // video slots
inline constexpr int kFrameStride = 5;      // camera frames between slots (0.5 s @ 10 Hz)
inline constexpr int kStatusDim = 6;
inline constexpr int kNumWaypoints = 8;

inline constexpr std::size_t kFrameNumel =
    3ull * kInputHeight * kInputWidth;                    // one CHW frame
inline constexpr std::size_t kVideoNumel =
    kFrameNumel * kNumHistFrames;                         // [3,4,256,512]

inline constexpr std::array<float, 3> kImageNetMean{0.485f, 0.456f, 0.406f};
inline constexpr std::array<float, 3> kImageNetStd{0.229f, 0.224f, 0.225f};

struct Waypoint {
  float x;        // forward [m], ego frame at inference time
  float y;        // left [m]
  float heading;  // [rad]
};

using Trajectory = std::array<Waypoint, kNumWaypoints>;  // t+0.5 .. t+4.0 s

// Ego status in physical units; encode() applies the training-time layout
// (subgoal is the route point 50 m of arc length ahead, divided by 10).
struct EgoStatus {
  float subgoal_x_m = 0.f;
  float subgoal_y_m = 0.f;
  float vel_x = 0.f;   // [m/s]
  float vel_y = 0.f;
  float acc_x = 0.f;   // [m/s^2]
  float acc_y = 0.f;

  std::array<float, kStatusDim> encode() const {
    return {subgoal_x_m / 10.f, subgoal_y_m / 10.f, vel_x, vel_y, acc_x, acc_y};
  }
};

}  // namespace latentdrive
