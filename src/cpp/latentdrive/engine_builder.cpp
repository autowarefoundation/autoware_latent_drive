#include "engine_builder.hpp"

#include <chrono>
#include <cstring>
#include <fstream>

#include "NvInfer.h"
#include "NvOnnxParser.h"
#include "logger.hpp"
#include "planner_params.hpp"
#include "utils.hpp"

namespace model {
namespace planner {

using namespace nvinfer1;

// @brief: This function is used to check a layer name against the fp32 pin
// list. The ONNX parser keeps the node scopes as layer names, so a prefix
// match reproduces what `trtexec --layerPrecisions='/planner/transformer/*'`
// matched on the command line
static bool IsPinnedFp32(const char *layer_name) {
  for (const char *prefix : kFp32LayerPrefixes) {
    if (std::strncmp(layer_name, prefix, std::strlen(prefix)) == 0) {
      return true;
    }
  }
  return false;
}

// @brief: This function is used to pin the planner-stage layers to fp32
// inside an fp16 build. Combined with kOBEY_PRECISION_CONSTRAINTS this is the
// C++ equivalent of the validated trtexec recipe (see build_engine.sh in the
// training repo)
// @brief: This function is used to check whether a layer produces indices or
// shapes (int/bool outputs, shape tensors). Those layers reject a Float
// precision constraint with an API usage error, so the pin skips them
static bool ComputesIndices(const ILayer &layer) {
  for (int o = 0; o < layer.getNbOutputs(); ++o) {
    const ITensor *tensor = layer.getOutput(o);
    if (tensor->isShapeTensor() || tensor->getType() == DataType::kINT32 ||
        tensor->getType() == DataType::kINT64 || tensor->getType() == DataType::kBOOL) {
      return true;
    }
  }
  return false;
}

static int PinPlannerLayersFp32(INetworkDefinition &network) {
  int pinned = 0;
  for (int i = 0; i < network.getNbLayers(); ++i) {
    ILayer *layer = network.getLayer(i);
    if (!IsPinnedFp32(layer->getName())) continue;
    // Constants have no compute precision to constrain, and index/shape
    // producers refuse a Float constraint outright
    if (layer->getType() == LayerType::kCONSTANT ||
        layer->getType() == LayerType::kSHAPE || ComputesIndices(*layer)) {
      continue;
    }
    layer->setPrecision(DataType::kFLOAT);
    ++pinned;
  }
  return pinned;
}

bool BuildEngine(const BuildParams &params) {
  auto logger = logger::CreateLogger(logger::Level::kINFO);

  // TensorRT 10: explicit batch is the only mode, the flag argument is gone
  auto builder = common::cuda::nv_unique_ptr<IBuilder>(createInferBuilder(*logger));
  auto network = common::cuda::nv_unique_ptr<INetworkDefinition>(builder->createNetworkV2(0));
  auto config  = common::cuda::nv_unique_ptr<IBuilderConfig>(builder->createBuilderConfig());
  auto parser  = common::cuda::nv_unique_ptr<nvonnxparser::IParser>(
      nvonnxparser::createParser(*network, *logger));
  if (!builder || !network || !config || !parser) {
    LOGE("Failed to create the TensorRT builder objects");
    return false;
  }

  config->setMemoryPoolLimit(MemoryPoolType::kWORKSPACE, params.workspace_size);

  if (!parser->parseFromFile(params.onnx_path.c_str(),
                             static_cast<int>(ILogger::Severity::kWARNING))) {
    LOGE("Failed to parse %s", params.onnx_path.c_str());
    return false;
  }
  LOG("Parsed %s (%d layers)", params.onnx_path.c_str(), network->getNbLayers());

  // Set the quantization flags (the TensorRT 8 platformHasFastFp16 probe is
  // deprecated in 10.x -- every supported GPU has fast fp16)
  if (params.prec == kFP16) {
    config->setFlag(BuilderFlag::kFP16);
    if (params.pin_planner_fp32) {
      config->setFlag(BuilderFlag::kOBEY_PRECISION_CONSTRAINTS);
      const int pinned = PinPlannerLayersFp32(*network);
      LOG("Pinned %d planner-stage layers to fp32 (mixed-precision recipe)", pinned);
    }
  }

  LOG("Building the %s engine, this takes a minute or two...",
      params.prec == kFP16 ? "fp16" : "fp32");
  const auto build_start = std::chrono::steady_clock::now();

  auto plan = common::cuda::nv_unique_ptr<IHostMemory>(
      builder->buildSerializedNetwork(*network, *config));
  if (!plan) {
    LOGE("buildSerializedNetwork failed");
    return false;
  }

  const double build_seconds =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - build_start).count();

  std::ofstream file(params.engine_path, std::ios::binary);
  if (!file) {
    LOGE("Cannot write %s", params.engine_path.c_str());
    return false;
  }
  file.write(static_cast<const char *>(plan->data()), plan->size());

  LOG("Engine built in %.1f s -> %s (%.0f MB)", build_seconds,
      params.engine_path.c_str(), plan->size() / 1e6);
  return true;
}

Precision GetPrecision(const std::string &prec) {
  if (prec == "fp32") return kFP32;
  if (prec != "fp16") {
    LOGW("Unknown precision '%s', falling back to fp16", prec.c_str());
  }
  return kFP16;
}

}  // namespace planner
}  // namespace model
