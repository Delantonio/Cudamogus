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

    for (int j = 1; j < blockDim.x; j *= 2)
    {
        int tmp = buffer[i - j];
        __syncthreads();

        if (i - blockidx * blockDim.x >= j)
            buffer[i] += tmp;
        __syncthreads();
    }
    
    int tmp = buffer[i - 1];
    __syncthreads();
    buffer[i] = tmp;

    if (threadIdx.x == 0)
    {
        buffer[i] = 0;
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
    

int main([[maybe_unused]] int argc, [[maybe_unused]] char* argv[])
{
    {
        // std::cout << "cub version : " << std::endl;
        //cub_main(argc, argv);
    }
    
    cpu_main(argc, argv);
    //gpu_main(argc, argv);

    return 0;
}