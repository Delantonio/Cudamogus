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
__global__ void kernel_shift(T* result, T* buffer, int size)
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
__global__
void kernel_inclusive_scan(T* buffer, T* scan_A, T* scan_P, int* blockstates, int* counter, int size) // scan_A + scan_P + blockstates are the same size (buffer_size / block_size)
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
        int tmp = buffer[i - j];
        __syncthreads();

        if (i - blockidx * blockDim.x >= j)
            buffer[i] += tmp;
        __syncthreads();
    }

    if (threadIdx.x == 0)
    {
        if (blockidx == 0)
        {
            atomicAdd(scan_P + blockidx, buffer[(blockidx+1) * blockDim.x - 1]);
            __threadfence_system();
            blockstates[blockidx] = 2;
        }
        else
        {
            atomicAdd(scan_A + blockidx, buffer[(blockidx+1) * blockDim.x - 1]);
            //__threadfence_system();
            blockstates[blockidx] = 1;
        }
    }
    __syncthreads();

    if (blockidx > 0)
    {
        //look back
        if (threadIdx.x == 0)
        {
            int idx = blockidx - 1;
            int state = atomicAdd(blockstates + idx, 0);
            //__threadfence_system();
            while (state != 2)
            {
                if (state == 1)
                {
                    int prev = atomicAdd(scan_A + idx, 0);
                    //__threadfence_system();
                    scan_P[blockidx] += prev;
                    idx--;
                }
                state = atomicAdd(blockstates + idx, 0);
                //__threadfence_system();
            }

            // prefix found
            int prevP = atomicAdd(scan_P + idx, 0);
            //__threadfence_system();
            int prevA = scan_A[idx];//atomicAdd(scan_A + idx, 0);
            //__threadfence_system();

            scan_P[blockidx] += prevP + prevA;
            blockstates[blockidx] = 2;
        }
        
        __syncthreads();
        buffer[i] += scan_P[blockidx];
    }
}

template <int BLOCK_SIZE>
__device__
void warp_reduce(int* sdata, int tid) {
    if (BLOCK_SIZE >= 64) {sdata[tid] += sdata[tid + 32]; __syncwarp(); }
    if (BLOCK_SIZE >= 32) {sdata[tid] += sdata[tid + 16]; __syncwarp(); }
    if (BLOCK_SIZE >= 16) {sdata[tid] += sdata[tid + 8]; __syncwarp(); }
    if (BLOCK_SIZE >= 8) {sdata[tid] += sdata[tid + 4]; __syncwarp(); }
    if (BLOCK_SIZE >= 4) {sdata[tid] += sdata[tid + 2]; __syncwarp(); }
    if (BLOCK_SIZE >= 2) {sdata[tid] += sdata[tid + 1]; __syncwarp(); }
}

template <typename T, int BLOCK_SIZE>
__global__
void kernel_reduce(const T* __restrict__ buffer, T* __restrict__ total, int size)
{
    extern __shared__ int sdata[];

    const unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * (blockDim.x * 2) + threadIdx.x;

    sdata[tid] = buffer[i] + buffer[i + blockDim.x];
    __syncthreads();

    if constexpr (BLOCK_SIZE >= 512) {
        if (tid < 256) { sdata[tid] += sdata[tid + 256]; } __syncthreads();
    }
    if constexpr (BLOCK_SIZE >= 256) {
        if (tid < 128) { sdata[tid] += sdata[tid + 128]; } __syncthreads();
    }
    if constexpr (BLOCK_SIZE >= 128) {
        if (tid < 64) { sdata[tid] += sdata[tid + 64]; } __syncthreads();
    }

    if (tid < 32)
        warp_reduce<BLOCK_SIZE>(sdata, tid);

    if (tid == 0) total[blockIdx.x] = sdata[0];
}

template <typename T>
__global__
void kernel_final_add(const T* __restrict__ buffer, T* __restrict__ total, int size)
{
    const int id = threadIdx.x + blockIdx.x * blockDim.x;
    if (id < size)
        atomicAdd(&total[0], buffer[id]);
}

void reduce(CudaArray1D<int> buffer, CudaArray1D<int> total)
{
    constexpr int blocksize = 512;
    const int gridsize = (buffer.size_ + blocksize - 1) / (blocksize * 2);

    int *tmp;
    cudaMalloc(&tmp, gridsize * sizeof(int));

	kernel_reduce<int, blocksize><<<gridsize, blocksize, blocksize * sizeof(int)>>>(buffer.data_, tmp, buffer.size_);
    kernel_final_add<int><<<gridsize / blocksize + 1, blocksize>>>(tmp, total.data_, gridsize);

    cudaDeviceSynchronize();
    cudaCheckError();
}

/*__device__ __host__ void print_debug(int *image_data, int size) 
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i == 0)
    {
        for (int i = 0; i < size; i++)
            printf("DEBUG: image_data[%d] = %d\n", i, image_data[i]);
    }
    __syncthreads();
}*/

template<typename T>
__global__ void build_predicate(T *image_data, T* predicate, int image_size)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= image_size)
        return;
    
    int garbage_value = -27;
    if (image_data[i] != garbage_value)
        predicate[i] = 1;
    
}

template<typename T>
__global__ void scatter_corresponding_adresses(T *image_data, T* image_data_backup, int* predicate, int image_size)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= image_size)
        return;
    
    int garbage_value = -27;
    if (image_data_backup[i] != garbage_value)
        image_data[predicate[i]] = image_data_backup[i];
}

template<typename T>
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
    /*if (image_data[i] < 0 || image_data[i] > 255)
        printf("map bug image_data[%d] = %d\n", i, image_data[i]);*/
    //if (i < 20)
    //    printf("apply_map_pixel image_data[%d] = %d\n", i, image_data[i]);
    /*if (i % 4 == 0)
        atomicAdd(image_data + i, 1);//image_data[i] += 1;
    else if (i % 4 == 1)
        atomicAdd(image_data + i, -5);//image_data[i] -= 5;
    else if (i % 4 == 2)
        atomicAdd(image_data + i, 3);//image_data[i] += 3;
    else if (i % 4 == 3)
        atomicAdd(image_data + i, -8);//image_data[i] -= 8;*/
}

template<typename T>
__global__ void kernel_histogram(T *image_data, int *histogram, int image_size)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= image_size)
        return;

    /*if (i == 0)
    {
        for (int j = 0; j < 20; j++)
        {
            printf("kernel_histogram image_data[%d] = %d\n", j, image_data[j]);
        }
    }
    
    __syncthreads();*/
    int image_value = image_data[i];
    /*if (image_value < 0 || image_value > 255)
        printf("histogram bug image_data[%d] = %d\n", i, image_value);*/
    atomicAdd(histogram + image_value, 1);
}

template<typename T>
__global__ void kernel_filter_zeros(T* histogram, int* predicate)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= 256)
        return;

    if (histogram[i] == 0)
    {
        predicate[i] = 0;
    }
    else
    {
        predicate[i] = 1;
    }
}

template<typename T>
__global__ void kernel_find_first_non_zero(T *histogram, int *summed_predicate, int* first_non_zero)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= 256)
        return;

    if (summed_predicate[i] == 1)
    {
        *first_non_zero = histogram[i];
    }
}

template<typename T>
__global__ void kernel_apply_map_transformation(T *result, T *image_data, int *histogram, int *first_non_zero, int image_size)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= image_size)
        return;

    result[i] = std::roundf(((histogram[image_data[i]] - *first_non_zero) / static_cast<float>(image_size - *first_non_zero)) * 255.0f);
}

// Mano
void fix_image_gpu(int *image_data, int *image_data_cpu, const int image_size, const int buffer_size)
{
    constexpr int blocksize = 256;
    int nb_blocks = buffer_size / blocksize;
    nb_blocks++;

    printf("image_size = %d, buffer_size = %d\n", image_size, buffer_size);

    int *predicate;
    cudaMalloc(&predicate, buffer_size * sizeof(int));
    cudaMemset(predicate, 0, buffer_size * sizeof(int));

    build_predicate<int><<<nb_blocks, blocksize>>>(image_data, predicate, buffer_size);
    cudaDeviceSynchronize();
    
    std::vector<int> predicate_cpu(buffer_size, 0);

    constexpr int garbage_val = -27;
    for (int i = 0; i < buffer_size; ++i)
        if (image_data_cpu[i] != garbage_val)
            predicate_cpu[i] = 1;

    int *predicate_gpu = (int*)malloc(sizeof(int) * buffer_size);
    cudaMemcpy(predicate_gpu, predicate, sizeof(int) * buffer_size, cudaMemcpyDeviceToHost);
    
    
    for (int i = 0; i < buffer_size; ++i)
    {
        if (predicate_cpu[i] != predicate_gpu[i])
        {
            printf("fail build predicate predicate[%d] = %d, predicate_gpu[%d] = %d\n", i, predicate_cpu[i], i, predicate_gpu[i]);
            exit(1);
        }
    }

    int* scan_A;
    cudaMalloc(&scan_A, nb_blocks * sizeof(int));
    cudaMemset(scan_A, 0, nb_blocks * sizeof(int));
    
    int* scan_P;
    cudaMalloc(&scan_P, nb_blocks * sizeof(int));
    cudaMemset(scan_P, 0, nb_blocks * sizeof(int));

    int* blockstates;
    cudaMalloc(&blockstates, nb_blocks * sizeof(int));
    cudaMemset(blockstates, 0, nb_blocks * sizeof(int)); // 0 = X; 1 == A; 2 == P

    int* counter;
    cudaMalloc(&counter, sizeof(int));
    cudaMemset(counter, 0, sizeof(int));

    // check if it is an exclusive scan
    kernel_inclusive_scan<int><<<nb_blocks, blocksize, sizeof(int)>>>(predicate, scan_A, scan_P, blockstates, counter, buffer_size);
    
    cudaDeviceSynchronize();

    int* shifted_predicate;
    cudaMalloc(&shifted_predicate, buffer_size * sizeof(int));
    cudaMemset(shifted_predicate, 0, buffer_size * sizeof(int));

    kernel_shift<int><<<nb_blocks, blocksize>>>(shifted_predicate, predicate, buffer_size);

    cudaDeviceSynchronize();

    std::exclusive_scan(predicate_cpu.begin(), predicate_cpu.end(), predicate_cpu.begin(), 0);
    
    cudaMemcpy(predicate_gpu, shifted_predicate, sizeof(int) * buffer_size, cudaMemcpyDeviceToHost);
    
    cudaDeviceSynchronize();
    
    for (int i = 0; i < buffer_size; ++i)
    {
        if (predicate_cpu[i] != predicate_gpu[i])
        {
            printf("scan fail predicate_cpu[%d] = %d, predicate_gpu[%d] = %d\n", i, predicate_cpu[i], i, predicate_gpu[i]);
            /*for (int j = i - 5; j < i + 5; j++)
            {
                printf("predicate_cpu[%d] = %d, predicate_gpu[%d] = %d\n", j, predicate_cpu[j], j, predicate_gpu[j]);
            }*/
            exit(1);
        }
    }

    int *image_data_copy;
    cudaMalloc(&image_data_copy, buffer_size * sizeof(int));
    cudaMemcpy(image_data_copy, image_data, buffer_size * sizeof(int), cudaMemcpyDeviceToDevice);
    
    scatter_corresponding_adresses<int><<<nb_blocks, blocksize>>>(image_data, image_data_copy, shifted_predicate, buffer_size);
    cudaDeviceSynchronize();
    
    int* image_data_gpu = (int*)malloc(sizeof(int) * buffer_size);
    cudaMemcpy(image_data_gpu, image_data, sizeof(int) * buffer_size, cudaMemcpyDeviceToHost);

    for (std::size_t i = 0; i < predicate_cpu.size(); ++i)
    {
        if (image_data_cpu[i] != garbage_val)
        {
            image_data_cpu[predicate_cpu[i]] = image_data_cpu[i];
        }
    }
    
    for (int i = 0; i < buffer_size; ++i)
    {
        if (image_data_cpu[i] != image_data_gpu[i])
        {
            printf("scatter fail image_cpu[%d] = %d, image_gpu[%d] = %d\n", i, image_data_cpu[i], i, image_data_gpu[i]);
            exit(1);
        }
    }

    apply_map_to_pixels<int><<<nb_blocks, blocksize>>>(image_data, image_size);
    cudaDeviceSynchronize();
    cudaMemcpy(image_data_gpu, image_data, sizeof(int) * image_size, cudaMemcpyDeviceToHost);

    for (int i = 0; i < image_size; ++i)
    {
        if (i % 4 == 0)
            image_data_cpu[i] += 1;
        else if (i % 4 == 1)
            image_data_cpu[i] -= 5;
        else if (i % 4 == 2)
            image_data_cpu[i] += 3;
        else if (i % 4 == 3)
            image_data_cpu[i] -= 8;
        if (image_data_cpu[i] < 0 || image_data_cpu[i] > 255)
            printf("cpu map bug image_data[%d] = %d\n", i, image_data_cpu[i]);
    }
    
    for (int i = 0; i < buffer_size; ++i)
    {
        if (image_data_cpu[i] != image_data_gpu[i])
        {
            printf("apply map fail image_cpu[%d] = %d, image_gpu[%d] = %d\n", i, image_data_cpu[i], i, image_data_gpu[i]);
            exit(1);
        }
    }

    // do histogram
    int* histogram;
    cudaMalloc(&histogram, 256 * sizeof(int));
    cudaMemset(histogram, 0, 256 * sizeof(int));

    kernel_histogram<int><<<nb_blocks, blocksize>>>(image_data, histogram, image_size);
    cudaDeviceSynchronize();
    
    int *histogram_gpu = (int*)malloc(sizeof(int) * 256);
    cudaMemcpy(histogram_gpu, histogram, sizeof(int) * 256, cudaMemcpyDeviceToHost);

    std::array<int, 256> histo;
    histo.fill(0);
    for (int i = 0; i < image_size; ++i)
    {
        if (image_data_cpu[i] < 0 || image_data_cpu[i] > 255)
            printf("histogram bug image_data[%d] = %d\n", i, image_data_cpu[i]);
        ++histo[image_data_cpu[i]];
    }
    for (int i = 0; i < 256; ++i)
    {
        if (histo[i] != histogram_gpu[i])
        {
            printf("histogram fail histo_cpu[%d] = %d, histogram_gpu[%d] = %d\n", i, histo[i], i, histogram_gpu[i]);
            exit(1);
        }
    }

    // do inclusive scan
    cudaMemset(scan_A, 0, nb_blocks * sizeof(int));
    cudaMemset(scan_P, 0, nb_blocks * sizeof(int));
    cudaMemset(blockstates, 0, nb_blocks * sizeof(int));
    cudaMemset(counter, 0, sizeof(int));
    kernel_inclusive_scan<int><<<1, blocksize, sizeof(int)>>>(histogram, scan_A, scan_P, blockstates, counter, 256);
    cudaDeviceSynchronize();
    
    cudaMemcpy(histogram_gpu, histogram, sizeof(int) * 256, cudaMemcpyDeviceToHost);

    std::inclusive_scan(histo.begin(), histo.end(), histo.begin());

    for (int i = 0; i < 256; ++i)
    {
        if (histo[i] != histogram_gpu[i])
        {
            printf("histogram inclusive scan fail histo_cpu[%d] = %d, histogram_gpu[%d] = %d\n", i, histo[i], i, histogram_gpu[i]);
            exit(1);
        }
    }
    
    //find first non zero
    int* first_non_zero;
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

    kernel_find_first_non_zero<int><<<1, blocksize>>>(histogram, predicate_find_first_non_zero, first_non_zero);
    cudaDeviceSynchronize();
    
    auto first_none_zero_cpu = std::find_if(histo.begin(), histo.end(), [](auto v) { return v != 0; });

    const int cdf_min = *first_none_zero_cpu;

    int *first_non_zero_gpu = (int*)malloc(sizeof(int));
    cudaMemcpy(first_non_zero_gpu, first_non_zero, sizeof(int), cudaMemcpyDeviceToHost);

    if (cdf_min != *first_non_zero_gpu)
    {
        printf("find first non zero fail cdf_min = %d, first_non_zero_gpu = %d\n", cdf_min, *first_non_zero_gpu);
        exit(1);
    }

    //Apply map transformation of the histogram equalization
    cudaMemcpy(image_data_copy, image_data, sizeof(int) * buffer_size, cudaMemcpyDeviceToDevice);

    kernel_apply_map_transformation<int><<<nb_blocks, blocksize>>>(image_data, image_data_copy, histogram, first_non_zero, image_size);
    cudaDeviceSynchronize();
    
    cudaMemcpy(image_data_gpu, image_data, sizeof(int) * image_size, cudaMemcpyDeviceToHost);
    
    std::transform(image_data_cpu, image_data_cpu + image_size, image_data_cpu,
        [image_size, cdf_min, &histo](int pixel)
            {
                return std::roundf(((histo[pixel] - cdf_min) / static_cast<float>(image_size - cdf_min)) * 255.0f);
            }
    );
    
    for (int i = 0; i < image_size; ++i)
    {
        if (image_data_cpu[i] != image_data_gpu[i])
        {
            printf("apply map transformation fail image_data_cpu[%d] = %d, image_data_gpu[%d] = %d\n", i, image_data_cpu[i], i, image_data_gpu[i]);
            exit(1);
        }
    }

    std::cout << "Done" << std::endl;
}

int gpu_main([[maybe_unused]] int argc, [[maybe_unused]] char* argv[])
{
    // -- Pipeline initialization

    std::cout << "File loading..." << std::endl;

    // - Get file paths

    using recursive_directory_iterator = std::filesystem::recursive_directory_iterator;
    std::vector<std::string> filepaths;
    for (const auto& dir_entry : recursive_directory_iterator("../images"))
        filepaths.emplace_back(dir_entry.path().string());

    // - Init pipeline object

    Pipeline pipeline(filepaths);

    // -- Main loop containing image retring from pipeline and fixing

    const int nb_images = pipeline.images.size();
    std::vector<Image> images(nb_images);

    // - One CPU thread is launched for each image

    std::cout << "Done, starting compute" << std::endl;

    //#pragma omp parallel for
    for (int i = 0; i < nb_images; ++i)
    {
        images[i] = pipeline.get_image(i);
        std::cout << "Image " << i << " loaded " << images[i].to_sort.id << std::endl;
        int *image_data;
        int image_size = images[i].width * images[i].height;
        cudaMalloc(&image_data, images[i].buffer.size() * sizeof(int));

        cudaMemcpy(image_data, images[i].buffer.data(), images[i].buffer.size() * sizeof(int), cudaMemcpyHostToDevice);
        fix_image_gpu(image_data, images[i].buffer.data(), image_size, images[i].buffer.size());

        std::cout << "Image " << i << " fixed " << images[i].to_sort.id << std::endl;

        cudaMemcpy(images[i].buffer.data(), image_data, image_size * sizeof(int), cudaMemcpyDeviceToHost);
        
        // for (int j = 50; j < image_size; j++)
        // {
        //     if (j == 80)
        //         break;
        //     std::cout << "DEBUG: image_data[" << j << "] = " << images[i].buffer[j] << std::endl;
        //     if (images[i].buffer[j] < 0 || images[i].buffer[j] > 255)
        //     {
        //         std::cout << "ERROR at "<< j << " : " << images[i].buffer[j] << " image_size: " << image_size << std::endl;
        //     }
        // }

        std::cout << "Image " << i << " copied " << images[i].to_sort.id << std::endl;
        
        std::ostringstream oss;
        oss << "ImageGPU#" << images[i].to_sort.id << ".pgm";
        std::string str = oss.str();
        images[i].write(str);
        std::cout << "Image " << i << " written" << std::endl;
    }

    return 0;
}


int main([[maybe_unused]] int argc, [[maybe_unused]] char* argv[])
{
    {
        // std::cout << "cub version : " << std::endl;
        //cub_main(argc, argv);
    }
    
    if (argc > 1)
        cpu_main(argc, argv);
    else
        gpu_main(argc, argv);

    return 0;
}