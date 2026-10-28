#ifndef __PLANNER_PARAMS_HPP__
#define __PLANNER_PARAMS_HPP__

#include "trajectory.hpp"

namespace model {
namespace planner {

/****************************************/
/***********Network related**************/
/****************************************/
// The exports share one contract and differ in three sizes:
//   single-view  video  float32 [B, 3, 4, 256, 512]
//   multi-view   video  float32 [B, V, 3, 4, 256, 512]   view 0 anchors the
//                                                         history (the front
//                                                         camera)
//   status float32 [B, 2K + 4]          K route points, then the motion terms
//   plan   float32 [B, N, 3]            (named `traj` in the older exports)
// B is 1 (older exports) or a symbolic axis the engine build freezes to 1.
// The view count V, the route point count K and the waypoint count N are read
// from the engine at load, so one binary serves every checkpoint:
//   V = 1  the CAM_FRONT models;  V = 2  wide + telephoto;
//   V = 3  camfront3: front, front_left, front_right
//   K = 1  the 50 m subgoal (status [B, 6]);  K = 6  route points (status [B, 16])
constexpr static char kVideoTensorName[]  = "video";
constexpr static char kStatusTensorName[] = "status";
constexpr static char kTrajTensorName[]   = "traj";  // legacy output name

constexpr static int kInputHeight   = 256;
constexpr static int kInputWidth    = 512;
constexpr static int kNumHistFrames = 4;  // video slots, oldest..newest, 0.5 s apart

// status = [p1x, p1y, ..., pKx, pKy, v_fwd, v_left, a_fwd, a_left]: each route
// point in the ego frame divided by kRoutePointDivisor, the motion terms in
// m/s and m/s^2 unscaled. kLegacyStatusDim is the single-subgoal contract
constexpr static int   kStatusMotionDim   = 4;
constexpr static int   kLegacyStatusDim   = 2 + kStatusMotionDim;
constexpr static float kRoutePointDivisor = 10.f;

constexpr static int kFrameNumel = 3 * kInputHeight * kInputWidth;  // one CHW frame
constexpr static int kVideoNumel = kFrameNumel * kNumHistFrames;    // one view: [3, 4, H, W]

// A multi-view tensor is V contiguous single-view blocks, [V, 3, 4, H, W];
// Planner::VideoNumel() is V * kVideoNumel for the loaded engine

/****************************************/
/**********Preprocess related************/
/****************************************/
// The history slots are kHistDtS apart in time. Camera frames run at 10 Hz,
// so on a healthy stream consecutive slots are kFrameStride frames apart --
// the count-based FrameWindow relies on that; the timestamped ViewWindow
// picks each slot by nearest stamp instead (planner_io.pick_history)
constexpr static double kHistDtS     = 0.5;
constexpr static int    kFrameStride = 5;

// ViewWindow tolerances. The wide history must reach back the full 1.5 s
// before a plan runs; kDefaultHistoryToleranceS is the slack on that span
// (half a 10 Hz period, so the warm-up ends with the 16th frame as the
// count-based window's does). The slots themselves are picked by nearest
// stamp with no tolerance, as planner_io.pick_history does. Every other view
// must have a frame within kDefaultViewToleranceS of the chosen view-0 frame:
// the camfront3 training paired the side cameras within 50 ms (the earlier
// two-view alignment used 60 ms). A slot without one skips the inference
// unless the window is told to fill it with the training's blank frame
constexpr static double kDefaultHistoryToleranceS = 0.05;
constexpr static double kDefaultViewToleranceS    = 0.05;

constexpr static float kImageNetMean[3] = {0.485f, 0.456f, 0.406f};
constexpr static float kImageNetStd[3]  = {0.229f, 0.224f, 0.225f};

/****************************************/
/*************Status related*************/
/****************************************/
// Route-point models (status [B, 16]) take the points this many metres of arc
// length past the ego's projection on a 1 m resampled route; the single-
// subgoal models take one point kLegacySubgoalAheadM ahead. Route vertices
// that turn back by more than kDefaultRouteReversalDeg are dropped first: a
// mission route that doubles back at a junction entry otherwise puts a point
// behind the car (the "route kinks" issue of the A/B model READMEs)
constexpr static double kDefaultRoutePointsM[] = {5.0, 10.0, 15.0, 20.0, 30.0, 50.0};
constexpr static double kLegacySubgoalAheadM    = 50.0;
constexpr static double kDefaultRouteReversalDeg = 120.0;
// The training differenced recorded positions for the motion terms
// (planner_io.ego_status): poses kMotionSampleDtS apart, each within
// kMotionSampleToleranceS of its target time
constexpr static double kMotionSampleDtS        = 0.1;
constexpr static double kMotionSampleToleranceS = 0.05;

/****************************************/
/*********Engine build related***********/
/****************************************/
// Layers pinned to fp32 in the fp16 build (the validated mixed-precision
// recipe). Prefixes match the ONNX node scopes of the planner stage --
// 713 of 11409 nodes, negligible compute next to the ViT encoder. The
// open-loop exports scope the model as `/planner/`, the closed-loop and
// multi-view ones (export_onnx.py in the training repo) as `/m/`; a V-view
// model has one image_fc per view (`image_fc`, `image_fc_1`, `image_fc_2`),
// so that prefix has no trailing slash and pins all of them.
// The build log says how many layers matched -- 0 means a new export scopes
// its nodes differently and this list needs the new prefix
constexpr static const char *kFp32LayerPrefixes[] = {
    "/planner/transformer/",
    "/planner/trajectory_head/",
    "/planner/image_fc/",
    "/planner/status_encoding/",
    "/m/transformer/",
    "/m/trajectory_head/",
    "/m/image_fc",
    "/m/status_encoding/",
};

}  // namespace planner
}  // namespace model

#endif  //__PLANNER_PARAMS_HPP__
