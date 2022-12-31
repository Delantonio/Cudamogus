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
__global__
void kernel_exclusive_scan(T* buffer, T* scan_A, T* scan_P, int* blockstates, int* counter, int size) // scan_A + scan_P + blockstates are the same size (buffer_size / block_size)
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

    // local scan TODO make it exclusive
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



template<typename T>
__global__ void build_predicate(T *image_data, T* predicate, int image_size)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= image_size)
        return;
    
    int garbage_value = -27;
    if (image_data[tid] != garbage_value)
        predicate[tid] = 1;
}

template<typename T>
__global__ void scatter_corresponding_adresses_and_apply_map_to_pixels(T *image_data, T* predicate, int image_size)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= image_size)
        return;
    
    int garbage_value = -27;
    if (image_data[i] != garbage_value)
        image_data[predicate[i]] = image_data[i];

    if (i % 4 == 0)
        image_data[i] += 1;
    else if (i % 4 == 1)
        image_data[i] -= 5;
    else if (i % 4 == 2)
        image_data[i] += 3;
    else if (i % 4 == 3)
        image_data[i] -= 8;
}

// Mano
void fix_image_gpu(int *image_data, int image_size)
{
    constexpr int blocksize = 256;
    int nb_blocks = image_size / blocksize;
    if (nb_blocks == 0)
    {
        nb_blocks = 1;
    } 

    int *predicate;
    cudaMalloc(&predicate, image_size * sizeof(int));
    cudaMemset(predicate, 0, image_size * sizeof(int));

    build_predicate<int><<<nb_blocks, blocksize>>>(image_data, predicate, image_size);
    
    cudaDeviceSynchronize();

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
    kernel_exclusive_scan<int><<<nb_blocks, blocksize, sizeof(int)>>>(predicate, scan_A, scan_P, blockstates, counter, image_size);

    cudaDeviceSynchronize();
    
    scatter_corresponding_adresses_and_apply_map_to_pixels<int><<<nb_blocks, blocksize>>>(image_data, predicate, image_size);

    // do histogram

    // do inclusive scan
    kernel_inclusive_scan<int><<<nb_blocks, blocksize, sizeof(int)>>>(predicate, scan_A, scan_P, blockstates, counter, image_size);

    //find first non zero

    //Apply map transformation of the histogram equalization
}
    
// Cub version
void fix_image(Image& to_fix)
{
    const int image_size = to_fix.width * to_fix.height;

    // #1 Compact

    // Build predicate vector

    std::vector<int> predicate(to_fix.buffer.size(), 0);

    constexpr int garbage_val = -27;
    for (std::size_t i = 0; i < to_fix.buffer.size(); ++i)
        if (to_fix.buffer[i] != garbage_val)
            predicate[i] = 1;

    // Compute the exclusive sum of the predicate

    int *d_es_input;
    cudaMalloc(&d_es_input, to_fix.buffer.size() * sizeof(int));
    cudaMemcpy(d_es_input, predicate.data(), to_fix.buffer.size() * sizeof(int),
               cudaMemcpyHostToDevice);

    int *d_es_scan; // exclusive sum scan
    cudaMalloc(&d_es_scan, to_fix.buffer.size() * sizeof(int));

    void *d_temp_storage = NULL;
    size_t temp_storage_bytes = 0;
    cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, d_es_input,
                                  d_es_scan, to_fix.buffer.size());
    cudaMalloc(&d_temp_storage, temp_storage_bytes);
    cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, d_es_input,
                                  d_es_scan, to_fix.buffer.size());

    cudaDeviceSynchronize();

    cudaMemcpy(predicate.data(), d_es_scan, to_fix.buffer.size() * sizeof(int),
               cudaMemcpyDeviceToHost);

    cudaFree(d_es_input);
    cudaFree(d_es_scan);
    
    // std::exclusive_scan(predicate.begin(), predicate.end(), predicate.begin(), 0);

    // Scatter to the corresponding addresses

    for (std::size_t i = 0; i < predicate.size(); ++i)
        if (to_fix.buffer[i] != garbage_val)
            to_fix.buffer[predicate[i]] = to_fix.buffer[i];


    // #2 Apply map to fix pixels

    for (int i = 0; i < image_size; ++i)
    {
        if (i % 4 == 0)
            to_fix.buffer[i] += 1;
        else if (i % 4 == 1)
            to_fix.buffer[i] -= 5;
        else if (i % 4 == 2)
            to_fix.buffer[i] += 3;
        else if (i % 4 == 3)
            to_fix.buffer[i] -= 8;
    }

    // #3 Histogram equalization

    // Histogram

    std::array<int, 256> histo;
    histo.fill(0);
    for (int i = 0; i < image_size; ++i)
        ++histo[to_fix.buffer[i]];

    // Compute the inclusive sum scan of the histogram

    int *d_is_input;
    cudaMalloc(&d_is_input, 256 * sizeof(int));
    cudaMemcpy(d_is_input, histo.data(), 256 * sizeof(int),
               cudaMemcpyHostToDevice);

    int *d_is_scan; // inclusive sum scan
    cudaMalloc(&d_is_scan, 256 * sizeof(int));

    void *d_is_temp_storage = NULL;
    size_t is_temp_storage_bytes = 0;
    cub::DeviceScan::InclusiveSum(d_is_temp_storage, is_temp_storage_bytes, d_is_input,
                                  d_is_scan, 256);
    cudaMalloc(&d_is_temp_storage, is_temp_storage_bytes);
    cub::DeviceScan::InclusiveSum(d_is_temp_storage, is_temp_storage_bytes, d_is_input,
                                  d_is_scan, 256);

    cudaDeviceSynchronize();

    cudaMemcpy(histo.data(), d_is_scan, 256 * sizeof(int),
               cudaMemcpyDeviceToHost);

    cudaFree(d_is_input);
    cudaFree(d_is_scan);
    // std::inclusive_scan(histo.begin(), histo.end(), histo.begin());

    // Find the first non-zero value in the cumulative histogram

    auto first_none_zero = std::find_if(histo.begin(), histo.end(), [](auto v) { return v != 0; });

    const int cdf_min = *first_none_zero;

    // Apply the map transformation of the histogram equalization

    std::transform(to_fix.buffer.data(), to_fix.buffer.data() + image_size, to_fix.buffer.data(),
        [image_size, cdf_min, &histo](int pixel)
            {
                return std::roundf(((histo[pixel] - cdf_min) / static_cast<float>(image_size - cdf_min)) * 255.0f);
            }
    );
}

// Mano main

// Cub main
int main([[maybe_unused]] int argc, [[maybe_unused]] char* argv[])
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

    #pragma omp parallel for
    for (int i = 0; i < nb_images; ++i)
    {
        // TODO : make it GPU compatible (aka faster)
        // You will need to copy images one by one on the GPU
        // You can store the images the way you want on the GPU
        // But you should treat the pipeline as a pipeline :
        // You *must not* copy all the images and only then do the computations
        // You must get the image from the pipeline as they arrive and launch computations right away
        // There are still ways to speeds this process of course (wait for last class)
        images[i] = pipeline.get_image(i);
        //fix_image_cpu(images[i]);
        fix_image(images[i]);
    }

    std::cout << "Done with compute, starting stats" << std::endl;

    // -- All images are now fixed : compute stats (total then sort)

    // - First compute the total of each image

    // TODO : make it GPU compatible (aka faster)
    // You can use multiple CPU threads for your GPU version using openmp or not
    // Up to you :)

    #pragma omp parallel for
    for (int i = 0; i < nb_images; ++i)
    {
        auto& image = images[i];
        const int image_size = image.width * image.height;
        //image.to_sort.total = std::reduce(image.buffer.cbegin(), image.buffer.cbegin() + image_size, 0);
        
        int *d_image;
        cudaMalloc(&d_image, image_size * sizeof(int));
        cudaMemcpy(d_image, image.buffer.data(), image_size * sizeof(int), cudaMemcpyHostToDevice);
        
        int *d_reduce;
        cudaMalloc(&d_reduce, image_size * sizeof(int));

        void     *d_temp_storage = NULL;
        size_t   temp_storage_bytes = 0; 
        cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes,
                           d_image, d_reduce, image_size);
        cudaMalloc(&d_temp_storage, temp_storage_bytes);
        cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes,
                           d_image, d_reduce, image_size);
        
        cudaDeviceSynchronize();
        
        cudaMemcpy(&image.to_sort.total, d_reduce, sizeof(int), cudaMemcpyDeviceToHost);
        
        cudaFree(d_image);
        cudaFree(d_reduce);
    }
    
    // - All totals are known, sort images accordingly (OPTIONAL)
    // Moving the actual images is too expensive, sort image indices instead
    // Copying to an id array and sort it instead

    // TODO OPTIONAL : for you GPU version you can store it the way you want
    // But just like the CPU version, moving the actual images while sorting will be too slow
    using ToSort = Image::ToSort;
    std::vector<ToSort> to_sort(nb_images);
    std::generate(to_sort.begin(), to_sort.end(), [n = 0, images] () mutable
    {
        return images[n++].to_sort;
    });

    // TODO OPTIONAL : make it GPU compatible (aka faster)
    std::sort(to_sort.begin(), to_sort.end(), [](ToSort a, ToSort b) {
        return a.total < b.total;
    });

    // TODO : Test here that you have the same results
    // You can compare visually and should compare image vectors values and "total" values
    // If you did the sorting, check that the ids are in the same order
    for (int i = 0; i < nb_images; ++i)
    {
        std::cout << "Image #" << images[i].to_sort.id << " total : " << images[i].to_sort.total << std::endl;
        std::ostringstream oss;
        oss << "Image#" << images[i].to_sort.id << ".pgm";
        std::string str = oss.str();
        images[i].write(str);
    }

    std::cout << "Done, the internet is safe now :)" << std::endl;

    return 0;
}

// Given Main
int cpu_main([[maybe_unused]] int argc, [[maybe_unused]] char* argv[])
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

    #pragma omp parallel for
    for (int i = 0; i < nb_images; ++i)
    {
        // TODO : make it GPU compatible (aka faster)
        // You will need to copy images one by one on the GPU
        // You can store the images the way you want on the GPU
        // But you should treat the pipeline as a pipeline :
        // You *must not* copy all the images and only then do the computations
        // You must get the image from the pipeline as they arrive and launch computations right away
        // There are still ways to speeds this process of course (wait for last class)
        images[i] = pipeline.get_image(i);
        fix_image_cpu(images[i]);
    }

    std::cout << "Done with compute, starting stats" << std::endl;

    // -- All images are now fixed : compute stats (total then sort)

    // - First compute the total of each image

    // TODO : make it GPU compatible (aka faster)
    // You can use multiple CPU threads for your GPU version using openmp or not
    // Up to you :)

    #pragma omp parallel for
    for (int i = 0; i < nb_images; ++i)
    {
        auto& image = images[i];
        const int image_size = image.width * image.height;
        image.to_sort.total = std::reduce(image.buffer.cbegin(), image.buffer.cbegin() + image_size, 0);
    }
    
    // - All totals are known, sort images accordingly (OPTIONAL)
    // Moving the actual images is too expensive, sort image indices instead
    // Copying to an id array and sort it instead

    // TODO OPTIONAL : for you GPU version you can store it the way you want
    // But just like the CPU version, moving the actual images while sorting will be too slow
    using ToSort = Image::ToSort;
    std::vector<ToSort> to_sort(nb_images);
    std::generate(to_sort.begin(), to_sort.end(), [n = 0, images] () mutable
    {
        return images[n++].to_sort;
    });

    // TODO OPTIONAL : make it GPU compatible (aka faster)
    std::sort(to_sort.begin(), to_sort.end(), [](ToSort a, ToSort b) {
        return a.total < b.total;
    });

    // TODO : Test here that you have the same results
    // You can compare visually and should compare image vectors values and "total" values
    // If you did the sorting, check that the ids are in the same order
    for (int i = 0; i < nb_images; ++i)
    {
        std::cout << "Image #" << images[i].to_sort.id << " total : " << images[i].to_sort.total << std::endl;
        std::ostringstream oss;
        oss << "Image#" << images[i].to_sort.id << ".pgm";
        std::string str = oss.str();
        images[i].write(str);
    }

    std::cout << "Done, the internet is safe now :)" << std::endl;

    return 0;
}
