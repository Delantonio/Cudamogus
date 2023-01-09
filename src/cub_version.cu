#include "cub_version.cuh"
#include "cuda_utils.cuh"
#include "kernels.cuh"

struct NotEqual
{
    int compare;
    CUB_RUNTIME_FUNCTION __forceinline__ NotEqual(int compare)
        : compare(compare)
    {
    }
    CUB_RUNTIME_FUNCTION __forceinline__ bool operator()(const int &a) const
    {
        return (a != compare);
    }
};

// Cub version
void fix_image(Image &to_fix)
{
    const int image_size = to_fix.width * to_fix.height;

    // #1 Compact

    constexpr int garbage_val = -27;

    CudaArray1D<int> d_in(to_fix.buffer.size());
    d_in.copy_from(to_fix.buffer.data(), cudaMemcpyHostToDevice);

    CudaArray1D<int> d_image_buffer_fixed(image_size);
    CudaArray1D<int> d_num_selected_out(1);

    NotEqual is_not_garbage(garbage_val);

    void *d_temp_storage = NULL;
    size_t temp_storage_bytes = 0;
    cub::DeviceSelect::If(d_temp_storage, temp_storage_bytes, d_in.data_, d_image_buffer_fixed.data_,
                          d_num_selected_out.data_, to_fix.buffer.size(), is_not_garbage);
    cudaMalloc(&d_temp_storage, temp_storage_bytes);
    cub::DeviceSelect::If(d_temp_storage, temp_storage_bytes, d_in.data_, d_image_buffer_fixed.data_,
                          d_num_selected_out.data_, to_fix.buffer.size(), is_not_garbage);
    cudaDeviceSynchronize();

    d_in.free();
    d_num_selected_out.free();
    cudaFree(d_temp_storage);

    // #2 Apply map to fix pixels
    int blocksize = 256;
    int nb_blocks = (image_size + blocksize - 1) / blocksize;

    apply_map_to_pixels<<<nb_blocks, blocksize>>>(d_image_buffer_fixed.data_, image_size);
    cudaDeviceSynchronize();

    // #3 Histogram equalization

    CudaArray1D<int> d_histo(256, 0);

    void *d_temp_storage_histo = NULL;
    size_t temp_storage_bytes_histo = 0;
    cub::DeviceHistogram::HistogramEven(d_temp_storage_histo, temp_storage_bytes_histo,
                                        d_image_buffer_fixed.data_, d_histo.data_, 257, 0, 256, image_size);
    cudaMalloc(&d_temp_storage_histo, temp_storage_bytes_histo);
    cub::DeviceHistogram::HistogramEven(d_temp_storage_histo, temp_storage_bytes_histo,
                                        d_image_buffer_fixed.data_, d_histo.data_, 257, 0, 256, image_size);

    cudaDeviceSynchronize();

    cudaFree(d_temp_storage_histo);

    // Computed d_histo is reused in the next step so no need to copy it back to host nor free it

    // Compute the inclusive sum scan of the histogram

    CudaArray1D<int> d_is_scan(256);

    void *d_is_temp_storage = NULL;
    size_t is_temp_storage_bytes = 0;
    cub::DeviceScan::InclusiveSum(d_is_temp_storage, is_temp_storage_bytes, d_histo.data_,
                                  d_is_scan.data_, 256);
    cudaMalloc(&d_is_temp_storage, is_temp_storage_bytes);
    cub::DeviceScan::InclusiveSum(d_is_temp_storage, is_temp_storage_bytes, d_histo.data_,
                                  d_is_scan.data_, 256);

    cudaDeviceSynchronize();

    cudaFree(d_is_temp_storage);

    d_histo.free();

    // Find the first non-zero value in the cumulative histogram

    blocksize = 768;
    nb_blocks = (image_size + blocksize - 1) / blocksize;

    CudaArray1D<int> first_non_zero(1, 0);

    CudaArray1D<int> predicate_find_first_non_zero(256, 0);

    kernel_filter_zeros<<<1, 256>>>(d_is_scan.data_, predicate_find_first_non_zero.data_);
    cudaDeviceSynchronize();

    CudaArray1D<int> scan_A(nb_blocks, 0);
    CudaArray1D<int> scan_P(nb_blocks, 0);
    CudaArray1D<int> blockstates(nb_blocks, 0);
    CudaArray1D<int> counter(1, 0);

    kernel_inclusive_scan<<<1, 256, sizeof(int)>>>(predicate_find_first_non_zero.data_, scan_A.data_, scan_P.data_, blockstates.data_, counter.data_, 256);
    cudaDeviceSynchronize();

    scan_A.free();
    scan_P.free();
    blockstates.free();
    counter.free();

    kernel_find_first_non_zero<<<1, 256>>>(d_is_scan.data_, predicate_find_first_non_zero.data_, first_non_zero.data_);
    cudaDeviceSynchronize();
    
    predicate_find_first_non_zero.free();
    
    CudaArray1D<int> result(to_fix.buffer.size(), 0);

    blocksize = 256;
    nb_blocks = (image_size + blocksize - 1) / blocksize;

    kernel_apply_map_transformation<<<nb_blocks, blocksize>>>(result.data_, d_image_buffer_fixed.data_, d_is_scan.data_, first_non_zero.data_, image_size);
    cudaDeviceSynchronize();
    
    result.copy_to(to_fix.buffer.data(), cudaMemcpyDeviceToHost);

    to_fix.to_sort.total = compute_statistics_cub(result, image_size);

    result.free();
    first_non_zero.free();
}

// Cub main
int cub_main([[maybe_unused]] int argc, [[maybe_unused]] char *argv[], Pipeline &pipeline, bool write_images)
{
    // -- Main loop containing image retring from pipeline and fixing

    const int nb_images = pipeline.images.size();
    std::vector<Image> images(nb_images);

    // - One CPU thread is launched for each image
    std::cout << "Starting compute" << std::endl;

#pragma omp parallel for
    for (int i = 0; i < nb_images; ++i)
    {
        // You will need to copy images one by one on the GPU
        // You can store the images the way you want on the GPU
        // But you should treat the pipeline as a pipeline :
        // You *must not* copy all the images and only then do the computations
        // You must get the image from the pipeline as they arrive and launch computations right away
        // There are still ways to speeds this process of course (wait for last class)
        images[i] = pipeline.get_image(i);
        fix_image(images[i]);
    }

    std::cout << "Done with compute, starting stats" << std::endl;

    // -- All images are now fixed : compute stats (total then sort)

    // - All totals are known, sort images accordingly (OPTIONAL)
    // Moving the actual images is too expensive, sort image indices instead
    // Copying to an id array and sort it instead

    auto image_totals = std::vector<int>(nb_images);
    std::transform(images.cbegin(), images.cend(), image_totals.begin(), [](const auto &image)
                   { return image.to_sort.total; });
    auto image_indices = std::vector<int>(nb_images);
    std::transform(images.cbegin(), images.cend(), image_indices.begin(), [](const auto &image)
                   { return image.to_sort.id; });

    CudaArray1D<int> d_keys_in(nb_images);
    CudaArray1D<int> d_keys_out(nb_images);
    CudaArray1D<int> d_values_in(nb_images);
    CudaArray1D<int> d_values_out(nb_images);

    d_keys_in.copy_from(image_totals.data(), cudaMemcpyHostToDevice);
    d_values_in.copy_from(image_indices.data(), cudaMemcpyHostToDevice);

    void *d_temp_storage = NULL;
    size_t temp_storage_bytes = 0;
    cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes,
                                    d_keys_in.data_, d_keys_out.data_,
                                    d_values_in.data_, d_values_out.data_,
                                    nb_images);
    cudaMalloc(&d_temp_storage, temp_storage_bytes);
    cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes,
                                    d_keys_in.data_, d_keys_out.data_,
                                    d_values_in.data_, d_values_out.data_,
                                    nb_images);
    cudaDeviceSynchronize();

    d_keys_out.copy_to(image_totals.data(), cudaMemcpyDeviceToHost);
    d_values_out.copy_to(image_indices.data(), cudaMemcpyDeviceToHost);

    d_keys_in.free();
    d_keys_out.free();
    d_values_in.free();
    d_values_out.free();
    cudaFree(d_temp_storage);

    // // - Print sorted images
    // std::cout << "Done with stats, starting output" << std::endl;
    // for (int i = 0; i < nb_images; ++i)
    // {
    //     // to align the output
    //     std::string s = image_indices[i] < 10 ? "0" : "";
    //     std::cout << "# Pre Sorting - Image " << s << image_indices[i] << " : " << image_totals[i] << std::endl;
    // }

    // TODO OPTIONAL : for you GPU version you can store it the way you want
    // But just like the CPU version, moving the actual images while sorting will be too slow
    using ToSort = Image::ToSort;
    std::vector<ToSort> to_sort(nb_images);
    std::generate(to_sort.begin(), to_sort.end(), [n = 0, images]() mutable
                  { return images[n++].to_sort; });

    // TODO OPTIONAL : make it GPU compatible (aka faster)
    std::sort(to_sort.begin(), to_sort.end(), [](ToSort a, ToSort b)
              { return a.total < b.total; });

    // TODO : Test here that you have the same results
    // You can compare visually and should compare image vectors values and "total" values
    if (write_images)
    {
        for (int i = 0; i < nb_images; ++i)
        {
            // If you did the sorting, check that the ids are in the same order
            assert(to_sort[i].id == image_indices[i]);

            std::string s = images[i].to_sort.id < 10 ? "0" : "";
            std::cout << "Image #" << s << images[i].to_sort.id << " total : " << images[i].to_sort.total << std::endl;
            std::ostringstream oss;
            oss << "Image#" << images[i].to_sort.id << ".pgm";
            std::string str = oss.str();
            images[i].write(str);
        }
    }

    pipeline.upload_images(images);

    std::cout << "Done, the internet is safe now :)" << std::endl;

    return 0;
}

uint64_t compute_statistics_cub(CudaArray1D<int> &image, int image_size)
{
    CudaArray1D<int> d_reduce(1);

    void *d_temp_storage = NULL;
    size_t temp_storage_bytes = 0;
    cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes,
                           image.data_, d_reduce.data_, image_size);
    cudaMalloc(&d_temp_storage, temp_storage_bytes);
    cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes,
                           image.data_, d_reduce.data_, image_size);

    cudaDeviceSynchronize();

    uint64_t ret = 0;
    d_reduce.copy_to((int *)&ret, cudaMemcpyDeviceToHost);

    d_reduce.free();
    cudaFree(d_temp_storage);

    return ret;
}