#ifndef __ENGINE_BUILDER_HPP__
#define __ENGINE_BUILDER_HPP__

#include <cstddef>
#include <string>

namespace model {
namespace planner {

enum Precision { kFP32,
                 kFP16 };

struct BuildParams {
  std::string onnx_path;
  std::string engine_path;
  Precision   prec           = kFP16;
  std::size_t workspace_size = 8ULL << 30;

  // The validated precision recipe: the ViT encoder carries all the compute
  // and runs fp16, while the small planner stage (transformer + heads, see
  // kFp32LayerPrefixes) is pinned to fp32. Pure fp16 put one golden sample
  // 0.4 mm over the 0.10 m tolerance; the pin brings it back for +0.2 ms.
  // Only consulted when prec == kFP16
  bool pin_planner_fp32 = true;
};

// @brief: This function is used to build a serialized engine from the ONNX
// export and save it to `engine_path`. TensorRT 10 API: explicit batch is
// implicit, the plan comes from buildSerializedNetwork, the workspace from
// setMemoryPoolLimit. Returns false when parsing or building fails
bool BuildEngine(const BuildParams &params);

// @brief: This function is used to map a --prec style string ("fp32", "fp16")
// to the Precision enum; unknown strings fall back to kFP16 with a warning
Precision GetPrecision(const std::string &prec);

}  // namespace planner
}  // namespace model

#endif  //__ENGINE_BUILDER_HPP__
