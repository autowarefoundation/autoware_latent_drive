#ifndef __PLANNER_PARAMS_HPP__
#define __PLANNER_PARAMS_HPP__

#include "trajectory.hpp"

namespace model {
namespace planner {

/****************************************/
/***********Network related**************/
/****************************************/
// The engine is exported with fixed shapes (batch 1, single camera stream):
//   video  float32 [1, 3, 4, 256, 512]
//   status float32 [1, 6]
//   traj   float32 [1, 8, 3]
constexpr static char kVideoTensorName[]  = "video";
constexpr static char kStatusTensorName[] = "status";
constexpr static char kTrajTensorName[]   = "traj";

constexpr static int kInputHeight   = 256;
constexpr static int kInputWidth    = 512;
constexpr static int kNumHistFrames = 4;  // video slots, oldest..newest, 0.5 s apart
constexpr static int kStatusDim     = 6;

constexpr static int kFrameNumel  = 3 * kInputHeight * kInputWidth;  // one CHW frame
constexpr static int kVideoNumel  = kFrameNumel * kNumHistFrames;    // [3, 4, H, W]
constexpr static int kOutputNumel = common::kNumWaypoints * 3;       // [8, 3]

/****************************************/
/**********Preprocess related************/
/****************************************/
// Camera frames run at 10 Hz but the video slots are 0.5 s apart, so
// consecutive slots are kFrameStride camera frames apart
constexpr static int kFrameStride = 5;

constexpr static float kImageNetMean[3] = {0.485f, 0.456f, 0.406f};
constexpr static float kImageNetStd[3]  = {0.229f, 0.224f, 0.225f};

}  // namespace planner
}  // namespace model

#endif  //__PLANNER_PARAMS_HPP__
