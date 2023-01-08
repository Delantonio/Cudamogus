#include "cub_version.cuh"
#include "baseline.hh"
#include "image.hh"
#include "pipeline.hh"
#include "fix_cpu.hh"
#include "cuda_utils.cuh"

#include <cub/cub.cuh>
#include <vector>
#include <iostream>
#include <algorithm>
#include <sstream>
#include <filesystem>
#include <numeric>

template <typename T>
__global__ void kernel_shift(T *result, T *buffer, int size)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= size)
    {
        return;
    }

    if (i == 0)
    {
        result[i] = 0;
    }
    else
    {
        result[i] = buffer[i - 1];
    }
}

template <typename T>
__global__ void kernel_inclusive_scan(T *buffer, T *scan_A, T *scan_P, int *blockstates, int *counter, int size) // scan_A + scan_P + blockstates are the same size (buffer_size / block_size)
{
    __shared__ int blockidx;

    if (threadIdx.x == 0)
    {
        blockidx = atomicAdd(counter, 1);
    }
    __syncthreads();

    int i = threadIdx.x + blockidx * blockDim.x;

    if (i >= size)
    {
        return;
    }

    // local scan
    for (int j = 1; j < blockDim.x; j *= 2)
    {
        int tmp = 0;
        if (i - j >= 0)
            tmp = buffer[i - j];
        __syncthreads();

        if (i - blockidx * blockDim.x >= j)
            buffer[i] += tmp;
        __syncthreads();
    }

    if (threadIdx.x == 0)
    {
        int last_index = (blockidx + 1) * blockDim.x - 1;
        if (last_index >= size)
        {
            last_index = size - 1;
        }

        if (blockidx == 0)
            atomicAdd(scan_P + blockidx, buffer[last_index]);
        else
            atomicAdd(scan_A + blockidx, buffer[last_index]);

        __threadfence_system();

        blockstates[blockidx] = 1 + (blockidx == 0);
    }

    __syncthreads();

    if (blockidx > 0)
    {
        // look back
        if (threadIdx.x == 0)
        {
            int idx = blockidx - 1;
            int state = atomicAdd(blockstates + idx, 0);

            while (state != 2)
            {
                if (state == 1)
                {
                    int prev = atomicAdd(scan_A + idx, 0);

                    scan_P[blockidx] += prev;
                    idx--;
                }
                state = atomicAdd(blockstates + idx, 0);
            }

            __threadfence_system();

            // prefix found
            int prevP = atomicAdd(scan_P + idx, 0);
            int prevA = atomicAdd(scan_A + idx, 0);

            scan_P[blockidx] += prevP + prevA;
            blockstates[blockidx] = 2;
        }

        __syncthreads();

        int value = atomicAdd(scan_P + blockidx, 0);
        buffer[i] += value;
    }
}

__inline__ __device__ int warp_reduce(int val)
{
#pragma unroll
    for (int offset = warpSize / 2; offset > 0; offset /= 2)
        val += __shfl_down_sync(~0, val, offset);
    return val;
}

__global__ void kernel_reduce(const int *__restrict__ buffer, int *__restrict__ total, int size)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    int val = 0;
    while (i < size)
    {
        val += buffer[i];
        i += blockDim.x * gridDim.x;
    }
    if (blockIdx.x * (blockDim.x * 2) + threadIdx.x < size)
        val = warp_reduce(val);

    if (threadIdx.x % warpSize == 0)
        atomicAdd(total, val);
}

__global__ void print_debug(int *image_data, int size)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i == 0)
    {
        for (int i = 0; i < size; i++)
            printf("DEBUG: image_data[%d] = %d\n", i, image_data[i]);
    }
    __syncthreads();
}

template <typename T>
__global__ void build_predicate(T *image_data, T *predicate, int image_size)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= image_size)
        return;

    int garbage_value = -27;
    predicate[i] += (image_data[i] != garbage_value);
    // if (image_data[i] != garbage_value)
    //     predicate[i] = 1;
}

template <typename T>
__global__ void scatter_corresponding_adresses(T *image_data, T *image_data_backup, int *predicate, int image_size)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= image_size)
        return;

    int garbage_value = -27;
    if (image_data_backup[i] != garbage_value)
        image_data[predicate[i]] = image_data_backup[i];
}

template <typename T>
__global__ void apply_map_to_pixels(T *image_data, int image_size)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= image_size)
        return;

    if (i % 4 == 0)
    {
        image_data[i] += 1;
    }
    else if (i % 4 == 1)
    {
        image_data[i] -= 5;
    }
    else if (i % 4 == 2)
    {
        image_data[i] += 3;
    }
    else if (i % 4 == 3)
    {
        image_data[i] -= 8;
    }
}

template <typename T>
__global__ void kernel_histogram(T *image_data, int *histogram, int image_size)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= image_size)
        return;

    int image_value = image_data[i];
    atomicAdd(histogram + image_value, 1);
    // histogram[image_value] += 1;
}

/*
template<typename T>
__global__ void kernel_histogram(T *image_data, int *histogram, int image_size) // marche pas
{
    constexpr int hist_size = ((1 << sizeof(T) * 8));
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    extern __shared__ int s_histo[];

    for (int i = threadIdx.x; i < hist_size; i += blockDim.x)
        s_histo[i] = 0;
    __syncthreads();

    if (i < image_size)
    {
        atomicAdd(s_histo + image_data[i], 1);
    }
    __syncthreads();

    for (int i = threadIdx.x; i < hist_size; i += blockDim.x)
        atomicAdd(histogram + i, s_histo[i]);
}*/

template <typename T>
__global__ void kernel_filter_zeros(T *histogram, int *predicate)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= 256)
        return;

    predicate[i] = histogram[i] != 0;
    /*if (histogram[i] == 0)
    {
        predicate[i] = 0;
    }
    else
    {
        predicate[i] = 1;
    }*/
}

template <typename T>
__global__ void kernel_find_first_non_zero(T *histogram, int *summed_predicate, int *first_non_zero)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= 256)
        return;

    if (summed_predicate[i] == 1)
    {
        *first_non_zero = histogram[i];
    }
}

template <typename T>
__global__ void kernel_apply_map_transformation(T *result, T *image_data, int *histogram, int *first_non_zero, int image_size)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= image_size)
        return;
    result[i] = std::roundf(((histogram[image_data[i]] - *first_non_zero) / static_cast<float>(image_size - *first_non_zero)) * 255.0f);
}

void fix_image_gpu(int *image_data, const int image_size, const int buffer_size)
{
    constexpr int blocksize = 256;
    int nb_blocks = buffer_size / blocksize;
    nb_blocks++;

    int *predicate;
    cudaMalloc(&predicate, buffer_size * sizeof(int));
    cudaMemset(predicate, 0, buffer_size * sizeof(int));

    build_predicate<int><<<nb_blocks, blocksize>>>(image_data, predicate, buffer_size);
    cudaDeviceSynchronize();

    int *scan_A;
    cudaMalloc(&scan_A, nb_blocks * sizeof(int));
    cudaMemset(scan_A, 0, nb_blocks * sizeof(int));

    int *scan_P;
    cudaMalloc(&scan_P, nb_blocks * sizeof(int));
    cudaMemset(scan_P, 0, nb_blocks * sizeof(int));

    int *blockstates;
    cudaMalloc(&blockstates, nb_blocks * sizeof(int));
    cudaMemset(blockstates, 0, nb_blocks * sizeof(int)); // 0 = X; 1 == A; 2 == P

    int *counter;
    cudaMalloc(&counter, sizeof(int));
    cudaMemset(counter, 0, sizeof(int));

    kernel_inclusive_scan<int><<<nb_blocks, blocksize, sizeof(int)>>>(predicate, scan_A, scan_P, blockstates, counter, buffer_size);
    cudaDeviceSynchronize();

    int *shifted_predicate;
    cudaMalloc(&shifted_predicate, buffer_size * sizeof(int));
    cudaMemset(shifted_predicate, 0, buffer_size * sizeof(int));

    kernel_shift<int><<<nb_blocks, blocksize>>>(shifted_predicate, predicate, buffer_size);
    cudaDeviceSynchronize();

    cudaFree(predicate);

    int *image_data_copy;
    cudaMalloc(&image_data_copy, buffer_size * sizeof(int));
    cudaMemcpy(image_data_copy, image_data, buffer_size * sizeof(int), cudaMemcpyDeviceToDevice);

    scatter_corresponding_adresses<int><<<nb_blocks, blocksize>>>(image_data, image_data_copy, shifted_predicate, buffer_size);
    cudaDeviceSynchronize();
    cudaFree(shifted_predicate);

    apply_map_to_pixels<int><<<nb_blocks, blocksize>>>(image_data, image_size);
    cudaDeviceSynchronize();

    // do histogram
    int *histogram;
    cudaMalloc(&histogram, 256 * sizeof(int));
    cudaMemset(histogram, 0, 256 * sizeof(int));

    // kernel_histogram<int><<<nb_blocks, blocksize, blocksize * sizeof(int)>>>(image_data, histogram, image_size);
    kernel_histogram<int><<<nb_blocks, blocksize>>>(image_data, histogram, image_size);
    cudaDeviceSynchronize();

    // do inclusive scan
    cudaMemset(scan_A, 0, nb_blocks * sizeof(int));
    cudaMemset(scan_P, 0, nb_blocks * sizeof(int));
    cudaMemset(blockstates, 0, nb_blocks * sizeof(int));
    cudaMemset(counter, 0, sizeof(int));

    kernel_inclusive_scan<int><<<1, blocksize, sizeof(int)>>>(histogram, scan_A, scan_P, blockstates, counter, 256);
    cudaDeviceSynchronize();

    // find first non zero
    int *first_non_zero;
    cudaMalloc(&first_non_zero, sizeof(int));
    cudaMemset(first_non_zero, 0, sizeof(int));

    int *predicate_find_first_non_zero;
    cudaMalloc(&predicate_find_first_non_zero, 256 * sizeof(int));
    cudaMemset(predicate_find_first_non_zero, 0, 256 * sizeof(int));

    kernel_filter_zeros<int><<<1, blocksize>>>(histogram, predicate_find_first_non_zero);
    cudaDeviceSynchronize();

    cudaMemset(scan_A, 0, nb_blocks * sizeof(int));
    cudaMemset(scan_P, 0, nb_blocks * sizeof(int));
    cudaMemset(blockstates, 0, nb_blocks * sizeof(int));
    cudaMemset(counter, 0, sizeof(int));

    kernel_inclusive_scan<int><<<1, blocksize, sizeof(int)>>>(predicate_find_first_non_zero, scan_A, scan_P, blockstates, counter, 256);
    cudaDeviceSynchronize();

    cudaFree(scan_A);
    cudaFree(scan_P);
    cudaFree(blockstates);
    cudaFree(counter);

    kernel_find_first_non_zero<int><<<1, blocksize>>>(histogram, predicate_find_first_non_zero, first_non_zero);
    cudaDeviceSynchronize();
    cudaFree(predicate_find_first_non_zero);

    // Apply map transformation of the histogram equalization
    cudaMemcpy(image_data_copy, image_data, sizeof(int) * buffer_size, cudaMemcpyDeviceToDevice);

    kernel_apply_map_transformation<int><<<nb_blocks, blocksize>>>(image_data, image_data_copy, histogram, first_non_zero, image_size);
    cudaDeviceSynchronize();
    cudaFree(histogram);
    cudaFree(image_data_copy);
    cudaFree(first_non_zero);
}

int gpu_main([[maybe_unused]] int argc, [[maybe_unused]] char *argv[])
{
    // -- Pipeline initialization

    std::cout << "File loading..." << std::endl;

    // - Get file paths

    using recursive_directory_iterator = std::filesystem::recursive_directory_iterator;
    std::vector<std::string> filepaths;
    for (const auto &dir_entry : recursive_directory_iterator("../images"))
        filepaths.emplace_back(dir_entry.path().string());

    // - Init pipeline object

    Pipeline pipeline(filepaths);

    // -- Main loop containing image retring from pipeline and fixing

    const int nb_images = pipeline.images.size();
    // const int nb_images = 1;
    std::vector<Image> images(nb_images);

    // - One CPU thread is launched for each image

    std::cout << "Done, starting compute" << std::endl;

    // std::cout << "real nb_images: " << pipeline.images.size() << std::endl;

#pragma omp parallel for
    for (int i = 0; i < nb_images; ++i)
    {
        images[i] = pipeline.get_image(i);
        int *image_data;
        int image_size = images[i].width * images[i].height;
        cudaMalloc(&image_data, images[i].buffer.size() * sizeof(int));

        cudaMemcpy(image_data, images[i].buffer.data(), images[i].buffer.size() * sizeof(int), cudaMemcpyHostToDevice);
        // fix_image_gpu_compare_cpu(image_data, images[i].buffer.data(), image_size, images[i].buffer.size());
        fix_image_gpu(image_data, image_size, images[i].buffer.size());

        cudaMemcpy(images[i].buffer.data(), image_data, image_size * sizeof(int), cudaMemcpyDeviceToHost);
        cudaFree(image_data);
    }
    std::cout << "Done with compute, starting stats" << std::endl;

#pragma omp parallel for
    for (int i = 0; i < nb_images; ++i)
    {
        auto &image = images[i];
        const int image_size = image.width * image.height;
        // image.to_sort.total = std::reduce(image.buffer.cbegin(), image.buffer.cbegin() + image_size, 0);

        int *d_image;
        cudaMalloc(&d_image, image_size * sizeof(int));
        cudaMemcpy(d_image, image.buffer.data(), image_size * sizeof(int), cudaMemcpyHostToDevice);

        int *d_reduce;
        cudaMalloc(&d_reduce, 1 * sizeof(int));
        cudaMemset(d_reduce, 0, 1 * sizeof(int));

        constexpr int blocksize = 512;

        // Nb Blocks is not fixed : it should absolutely be modified and benchmarked to get the best performance
        const int nb_blocks = (image_size + blocksize - 1) / (blocksize);

        kernel_reduce<<<nb_blocks / 4, blocksize>>>(d_image, d_reduce, image_size);

        cudaDeviceSynchronize();

        cudaMemcpy(&image.to_sort.total, d_reduce, 1 * sizeof(int), cudaMemcpyDeviceToHost);

        cudaFree(d_image);
        cudaFree(d_reduce);

        std::string s = images[i].to_sort.id < 10 ? "0" : "";

        std::cout << "GPU - image.to_sort.total " << s << images[i].to_sort.id << ": " << image.to_sort.total << std::endl;

        image.to_sort.total = std::reduce(image.buffer.cbegin(), image.buffer.cbegin() + image_size, 0);

        std::cout << "CPU - image.to_sort.total " << s << images[i].to_sort.id << ": " << image.to_sort.total << std::endl;
    }

    using ToSort = Image::ToSort;
    std::vector<ToSort> to_sort(nb_images);
    std::generate(to_sort.begin(), to_sort.end(), [n = 0, images]() mutable
                  { return images[n++].to_sort; });
    std::sort(to_sort.begin(), to_sort.end(), [](ToSort a, ToSort b)
              { return a.total < b.total; });
    for (int i = 0; i < nb_images; ++i)
    {
        std::string s = images[i].to_sort.id < 10 ? "0" : "";
        std::cout << "Image #" << s << images[i].to_sort.id << " total : " << images[i].to_sort.total << std::endl;
        std::ostringstream oss;
        oss << "Image#" << images[i].to_sort.id << ".pgm";
        std::string str = oss.str();
        images[i].write(str);

        std::cout << "Image " << images[i].to_sort.id << " fixed\n"
                  << std::endl;
    }

    std::cout << "Done, the internet is safe now :)" << std::endl;

    return 0;
}

int main([[maybe_unused]] int argc, [[maybe_unused]] char *argv[])
{
    {
        // std::cout << "cub version : " << std::endl;
        // cub_main(argc, argv);
    }

    std::cout << "cpu main" << std::endl;
    cpu_main(argc, argv);

    std::cout << "gpu main" << std::endl;
    gpu_main(argc, argv);

    // if (argc > 1)
    //     cpu_main(argc, argv);
    // else
    //     gpu_main(argc, argv);

    return 0;
}