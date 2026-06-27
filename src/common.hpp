#pragma once
#include <cstdio>
#define VD_LOG(...)  do { std::fprintf(stderr, "[voicedetect] " __VA_ARGS__); std::fprintf(stderr, "\n"); } while (0)
