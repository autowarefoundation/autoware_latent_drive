#include "logger.hpp"

#include <stdio.h>
#include <stdlib.h>

namespace logger {

Level Logger::g_k_level = Level::kINFO;

Logger::Logger(Level level) : level_(level) {}

void Logger::LogInfo(Level level, const char *format, ...) {
  if (level > g_k_level) return;

  char    msg[MAXLOGSIZE];
  va_list args;
  va_start(args, format);
  vsnprintf(msg, sizeof(msg), format, args);
  va_end(args);

  switch (level) {
    case Level::kFATAL:
      fprintf(stderr, RED "[fatal]" CLEAR " %s\n", msg);
      exit(1);
    case Level::kERROR:
      fprintf(stderr, RED "[error]" CLEAR " %s\n", msg);
      break;
    case Level::kWARN:
      fprintf(stderr, YELLOW "[warn]" CLEAR " %s\n", msg);
      break;
    case Level::kINFO:
      fprintf(stdout, GREEN "[info]" CLEAR " %s\n", msg);
      break;
    case Level::kVERB:
      fprintf(stdout, PURPLE "[verb]" CLEAR " %s\n", msg);
      break;
    case Level::kDEBUG:
      fprintf(stdout, DGREEN "[debug]" CLEAR " %s\n", msg);
      break;
  }
}

Level Logger::GetLevel(Severity severity) {
  switch (severity) {
    case Severity::kINTERNAL_ERROR:
      return Level::kFATAL;
    case Severity::kERROR:
      return Level::kERROR;
    case Severity::kWARNING:
      return Level::kWARN;
    case Severity::kINFO:
      return Level::kINFO;
    default:
      return Level::kVERB;
  }
}

void Logger::log(Severity severity, const char *msg) noexcept {
  // Internal errors from TensorRT should not kill the process here; the
  // failed API call is checked at the call site, so cap at kERROR
  Level level = GetLevel(severity);
  if (level == Level::kFATAL) level = Level::kERROR;
  LogInfo(level, "[TRT] %s", msg);
}

void SetLogLevel(Level level) {
  Logger::g_k_level = level;
}

std::shared_ptr<Logger> CreateLogger(Level level) {
  return std::make_shared<Logger>(level);
}

};  // namespace logger
