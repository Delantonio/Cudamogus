#include "kernels.cuh"

__global__ void kernel_shift(int *result, int *buffer, int size)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;

    if (i < size)
    {
        result[i] = i == 0 ? 0 : buffer[i - 1];
    }
}

__global__ void kernel_inclusive_scan(int *buffer, int *scan_A, int *scan_P, int *blockstates, int *counter, int size) // scan_A + scan_P + blockstates are the same size (buffer_size / block_size)
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

__global__ void build_predicate(int *image_data, int *predicate, int image_size)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= image_size)
        return;

    int garbage_value = -27;
    predicate[i] += (image_data[i] != garbage_value);
}

__global__ void scatter_corresponding_adresses(int *image_data, int *image_data_backup, int *predicate, int image_size)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= image_size)
        return;

    int garbage_value = -27;
    if (image_data_backup[i] != garbage_value)
        image_data[predicate[i]] = image_data_backup[i];
}

__global__ void apply_map_to_pixels(int *image_data, int image_size)
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

__global__ void kernel_histo(int *image_data, int *histo, int N)
{
    constexpr int hist_size = 256;
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= N)
        return;
    extern __shared__ int s_histo[];

    // Block-loop pattern
    for (int i = threadIdx.x; i < hist_size; i += blockDim.x)
        s_histo[i] = 0;
    __syncthreads();

    if (i < N)
    {
        int image_value = image_data[i];
        atomicAdd(s_histo + image_value, 1);
    }
    __syncthreads();

    for (int i = threadIdx.x; i < hist_size; i += blockDim.x)
        atomicAdd(histo + i, s_histo[i]);
}

__global__ void kernel_filter_zeros(int *histogram, int *predicate)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= 256)
        return;

    predicate[i] = histogram[i] != 0;
}

__global__ void kernel_find_first_non_zero(int *histogram, int *summed_predicate, int *first_non_zero)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= 256)
        return;

    if (summed_predicate[i] == 1)
    {
        *first_non_zero = histogram[i];
    }
}

__global__ void kernel_apply_map_transformation(int *result, int *image_data, int *histogram, int *first_non_zero, int image_size)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= image_size)
        return;
    result[i] = std::roundf(((histogram[image_data[i]] - *first_non_zero) / static_cast<float>(image_size - *first_non_zero)) * 255.0f);
}
