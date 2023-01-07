#include "cub_version.cuh"
#include "cuda_utils.cuh"

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

// Cub main
int cub_main([[maybe_unused]] int argc, [[maybe_unused]] char* argv[])
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
    // const int nb_images = 1;
    std::vector<Image> images(nb_images);

    // - One CPU thread is launched for each image
    std::cout << "Done, starting compute" << std::endl;

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
        auto& image = images[i];
        const int image_size = image.width * image.height;
        //image.to_sort.total = std::reduce(image.buffer.cbegin(), image.buffer.cbegin() + image_size, 0);
        
        // std::cout << "first cuda array" << std::endl;
        // CudaArray1D<int> d_image(image.buffer.size(), image.buffer.data());
        // std::cout << "second cuda array" << std::endl;
        // CudaArray1D<int> d_reduce(1);
        
        // std::cout << "reduce start" << std::endl;
        // reduce(d_image, d_reduce);
        // cudaDeviceSynchronize();
        // cudaCheckError();
        
        // std::cout << "host array" << std::endl;
        // // auto truc = d_reduce.get_host_array();
        // // cudaCheckError();
        
        // cudaMemcpy(&image.to_sort.total, d_reduce.data_, 1 * sizeof(int), cudaMemcpyDeviceToHost);
        // cudaCheckError();
        
        // std::cout << "image.to_sort.total: " << image.to_sort.total << std::endl;
        
        // std::cout << "free d_image" << std::endl;
        // d_image.free();
        // std::cout << "free d_reduce" << std::endl;
        // d_reduce.free();
        
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
    
    auto image_totals = std::vector<int>(nb_images);
    std::transform(images.cbegin(), images.cend(), image_totals.begin(), [](const auto& image) { return image.to_sort.total; });
    auto image_indices = std::vector<int>(nb_images);
    std::transform(images.cbegin(), images.cend(), image_indices.begin(), [](const auto& image) { return image.to_sort.id; });

    // #pragma omp parallel for
    // for(int i = 0; i < nb_images; i++)
    // {
    //     image_indices[i] = to_sort[i].id;
    //     image_totals[i] = to_sort[i].total;
    // }

    int *d_keys_in;
    int *d_keys_out;
    int *d_values_in;
    int *d_values_out;

    cudaMalloc(&d_keys_in, nb_images * sizeof(int));
    cudaMalloc(&d_keys_out, nb_images * sizeof(int));
    cudaMalloc(&d_values_in, nb_images * sizeof(int));
    cudaMalloc(&d_values_out, nb_images * sizeof(int));

    cudaMemcpy(d_keys_in, image_totals.data(), nb_images * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_values_in, image_indices.data(), nb_images * sizeof(Image), cudaMemcpyHostToDevice);
    
    void     *d_temp_storage = NULL;
    size_t   temp_storage_bytes = 0;
    cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes,
                                d_keys_in, d_keys_out,
                                d_values_in, d_values_out,
                                nb_images);
    cudaMalloc(&d_temp_storage, temp_storage_bytes);
    cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes,
                                d_keys_in, d_keys_out,
                                d_values_in, d_values_out,
                                nb_images);
    cudaDeviceSynchronize();
    
    cudaMemcpy(image_totals.data(), d_keys_out, nb_images * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(image_indices.data(), d_values_out, nb_images * sizeof(Image), cudaMemcpyDeviceToHost);

    cudaFree(d_keys_in);
    cudaFree(d_keys_out);
    cudaFree(d_values_in);
    cudaFree(d_values_out);
    
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

    std::cout << "Done, the internet is safe now :)" << std::endl;

    return 0;
}