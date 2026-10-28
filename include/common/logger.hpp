#ifndef __LOGGER_HPP__
#define __LOGGER_HPP__

#include <stdarg.h>

#include <memory>

#include "NvInfer.h"

#define LOGD(...) logger::Logger::LogInfo(logger::Level::kDEBUG, __VA_ARGS__)
#define LOGV(...) logger::Logger::LogInfo(logger::Level::kVERB, __VA_ARGS__)
#define LOG(...) logger::Logger::LogInfo(logger::Level::kINFO, __VA_ARGS__)
#define LOGW(...) logger::Logger::LogInfo(logger::Level::kWARN, __VA_ARGS__)
#define LOGE(...) logger::Logger::LogInfo(logger::Level::kERROR, __VA_ARGS__)
#define LOGF(...) logger::Logger::LogInfo(logger::Level::kFATAL, __VA_ARGS__)

#define DGREEN "\033[1;36m"
#define BLUE "\033[1;34m"
#define PURPLE "\033[1;35m"
#define GREEN "\033[1;32m"
#define YELLOW "\033[1;33m"
#define RED "\033[1;31m"
#define CLEAR "\033[0m"

#define MAXLOGSIZE 10000

namespace logger {

enum class Level : int {
  kFATAL = 0,
  kERROR = 1,
  kWARN  = 2,
  kINFO  = 3,
  kVERB  = 4,
  kDEBUG = 5
};

// @brief This customize Logger class. Can be used to log information.
// Can be used as a macro or as a instance. The instance form also implements
// nvinfer1::ILogger, so TensorRT messages go through the same channel.
// The logging level can be modified dynamically by changing the g_k_level
class Logger : public nvinfer1::ILogger {
 public:
  Logger(){};
  explicit Logger(Level level);

  // @brief: This function is generate logging information depending on the
  // format. This function is set as a static function since it will be called
  // from macro, which won't create an object
  //
  // @param[in] level, the logging level for the function call
  // @param[in] format, the format for the logging message
  // @param[in] ..., variable arguments
  //
  // @return The formated logging message, and the logging level. Only output
  // the logging when the level is smaller than the pre-defined logging level.
  static void LogInfo(Level level, const char *format, ...);

  // @brief: This function is used to override the virtual function in ILogger
  void log(Severity severity, const char *msg) noexcept override;

  // @brief: This function is used to get Level from nv severity level
  static Level GetLevel(Severity severity);

  // The static variable to control every logger instance level
  static Level g_k_level;

 private:
  // The private variable to control the instance level
  Level level_ = Level::kINFO;
};

// @brief: This function is used to set logging level globally
void SetLogLevel(Level level);

// @brief: This function is used as a function to provide
// interface to generate Logger object
std::shared_ptr<Logger> CreateLogger(Level level);

};  // namespace logger

#endif  //__LOGGER_HPP__
