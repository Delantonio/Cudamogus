#include "cub_version.cuh"
#include "cuda_utils.cuh"

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

    // num_selected_out is the number of elements that are not garbage, so it should be image_size
    int num_selected_out;
    d_num_selected_out.copy_to(&num_selected_out, cudaMemcpyDeviceToHost);
    to_fix.buffer.resize(num_selected_out);
    d_image_buffer_fixed.copy_to(to_fix.buffer.data(), cudaMemcpyDeviceToHost);

    d_in.free();
    d_image_buffer_fixed.free();
    d_num_selected_out.free();
    cudaFree(d_temp_storage);

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

    CudaArray1D<int> d_samples(image_size);
    d_samples.copy_from(to_fix.buffer.data(), cudaMemcpyHostToDevice);

    CudaArray1D<int> d_histo(256, 0);

    void *d_temp_storage_histo = NULL;
    size_t temp_storage_bytes_histo = 0;
    cub::DeviceHistogram::HistogramEven(d_temp_storage_histo, temp_storage_bytes_histo,
                                        d_samples.data_, d_histo.data_, 257, 0, 256, image_size);
    cudaMalloc(&d_temp_storage_histo, temp_storage_bytes_histo);
    cub::DeviceHistogram::HistogramEven(d_temp_storage_histo, temp_storage_bytes_histo,
                                        d_samples.data_, d_histo.data_, 257, 0, 256, image_size);
    cudaDeviceSynchronize();

    d_samples.free();
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

    d_is_scan.copy_to(histo.data(), cudaMemcpyDeviceToHost);

    d_is_scan.free();
    cudaFree(d_is_temp_storage);

    d_histo.free();

    // Find the first non-zero value in the cumulative histogram

    auto first_none_zero = std::find_if(histo.begin(), histo.end(), [](auto v)
                                        { return v != 0; });

    const int cdf_min = *first_none_zero;

    // Apply the map transformation of the histogram equalization

    std::transform(to_fix.buffer.data(), to_fix.buffer.data() + image_size, to_fix.buffer.data(),
                   [image_size, cdf_min, &histo](int pixel)
                   {
                       return std::roundf(((histo[pixel] - cdf_min) / static_cast<float>(image_size - cdf_min)) * 255.0f);
                   });
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

    // - First compute the total of each image

#pragma omp parallel for
    for (int i = 0; i < nb_images; ++i)
    {
        auto &image = images[i];
        const int image_size = image.width * image.height;

        CudaArray1D<int> d_image(image_size);
        d_image.copy_from(image.buffer.data(), cudaMemcpyHostToDevice);

        CudaArray1D<int> d_reduce(1);

        void *d_temp_storage = NULL;
        size_t temp_storage_bytes = 0;
        cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes,
                               d_image.data_, d_reduce.data_, image_size);
        cudaMalloc(&d_temp_storage, temp_storage_bytes);
        cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes,
                               d_image.data_, d_reduce.data_, image_size);

        cudaDeviceSynchronize();

        d_reduce.copy_to((int*)&image.to_sort.total, cudaMemcpyDeviceToHost);

        d_image.free();
        d_reduce.free();
        cudaFree(d_temp_storage);
    }

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