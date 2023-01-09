#pragma once

#include "image.hh"
#include "pipeline.hh"
#include "cuda_utils.cuh"

#include <cub/cub.cuh>
#include <vector>
#include <iostream>
#include <algorithm>
#include <sstream>
#include <filesystem>
#include <numeric>

int cub_main([[maybe_unused]] int argc, [[maybe_unused]] char* argv[], Pipeline &pipeline);