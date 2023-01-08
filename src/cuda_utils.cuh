#pragma once

#define cudaCheckError()                                             \
    {                                                                \
        cudaError_t e = cudaGetLastError();                          \
        if (e != cudaSuccess)                                        \
        {                                                            \
            printf("Cuda failure %s:%d: '%s'\n", __FILE__, __LINE__, \
                   cudaGetErrorString(e));                           \
            exit(EXIT_FAILURE);                                      \
        }                                                            \
    }

template <typename T>
class CudaArray1D
{
public:
    T *data_; // The underlying GPU address to pass to a kernel
    const size_t size_;

    CudaArray1D(size_t size)
        : size_(size)
    {
        cudaMalloc(&data_, size_ * sizeof(T));
        cudaCheckError();
    }

    CudaArray1D(size_t size, T *array_host)
        : CudaArray1D(size)
    {
        cudaMemcpy(data_, array_host, size_ * sizeof(T),
                   cudaMemcpyHostToDevice);
        cudaCheckError();
    }

    ~CudaArray1D()
    {
        if (has_been_freed)
            return;
        free();
    }

    void free()
    {
        if (has_been_freed)
            throw std::runtime_error(
                "CudaArray1D : Trying to free already freed array");
        cudaFree(data_);
        cudaCheckError();
        has_been_freed = true;
    }

    T *get_host_array()
    {
        T *res = new T[size_];

        cudaMemcpy(res, data_, size_ * sizeof(T), cudaMemcpyDeviceToHost);
        cudaCheckError();
        return res;
    }

private:
    bool has_been_freed = false;
};