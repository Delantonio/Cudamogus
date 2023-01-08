#pragma once
#include <stdio.h>

__global__ void kernel_shift(int *result, int *buffer, int size);

__global__ void kernel_inclusive_scan(int *buffer, int *scan_A, int *scan_P, int *blockstates, int *counter, int size);

__global__ void kernel_reduce(const int *__restrict__ buffer, int *__restrict__ total, int size);

__global__ void print_debug(int *image_data, int size);

__global__ void build_predicate(int *image_data, int *predicate, int image_size);

__global__ void scatter_corresponding_adresses(int *image_data, int *image_data_backup, int *predicate, int image_size);

__global__ void apply_map_to_pixels(int *image_data, int image_size);

__global__ void kernel_histo(int *image_data, int *histo, int N);

__global__ void kernel_filter_zeros(int *histogram, int *predicate);

__global__ void kernel_find_first_non_zero(int *histogram, int *summed_predicate, int *first_non_zero);

__global__ void kernel_apply_map_transformation(int *result, int *image_data, int *histogram, int *first_non_zero, int image_size);
