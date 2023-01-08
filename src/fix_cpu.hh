#pragma once

#include "image.hh"
#include "pipeline.hh"

#include <vector>
#include <iostream>
#include <algorithm>
#include <sstream>
#include <filesystem>
#include <numeric>

int cpu_main([[maybe_unused]] int argc, [[maybe_unused]] char* argv[]);

void fix_image_cpu(Image& to_fix);
