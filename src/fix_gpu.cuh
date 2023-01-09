#pragma once

#include "image.hh"
#include "cuda_utils.cuh"
#include "pipeline.hh"

int gpu_main([[maybe_unused]] int argc, [[maybe_unused]] char *argv[], Pipeline &pipeline, bool write_images);

void fix_image_gpu(CudaArray1D<int> &image_data, const int image_size, const int buffer_size);
void compute_statistics(std::vector<Image> &images);
