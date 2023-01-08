#include "fix_gpu.cuh"

#include <iostream>
#include <filesystem>
#include <algorithm>

#include "kernels.cuh"
#include "pipeline.hh"


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

    const int nb_images = (int)pipeline.images.size();
    std::vector<Image> images(nb_images);

    std::cout << "Done, starting compute" << std::endl;

#pragma omp parallel for
    for (int i = 0; i < nb_images; ++i)
    {
        images[i] = pipeline.get_image(i);
        int image_size = (int)images[i].width * (int)images[i].height;

        CudaArray1D<int> image_data(images[i].buffer.size());
        image_data.copy_from(images[i].buffer.data(), cudaMemcpyHostToDevice);

        fix_image_gpu(image_data, image_size, (int)images[i].buffer.size());

        image_data.copy_to(images[i].buffer.data(), cudaMemcpyDeviceToHost);
        image_data.free();
    }
    std::cout << "Done with compute, starting stats" << std::endl;

    compute_statistics(images);

    return 0;
}

void fix_image_gpu(CudaArray1D<int> &image_data, const int image_size, const int buffer_size)
{
    constexpr int blocksize = 256;
    int nb_blocks = buffer_size / blocksize;
    nb_blocks++;

    CudaArray1D<int> predicate(buffer_size, 0);

    build_predicate<<<nb_blocks, blocksize>>>(image_data.data_, predicate.data_, buffer_size);
    cudaDeviceSynchronize();

    CudaArray1D<int> scan_A(nb_blocks, 0);
    CudaArray1D<int> scan_P(nb_blocks, 0);
    CudaArray1D<int> blockstates(nb_blocks, 0); // 0 = X; 1 == A; 2 == P
    CudaArray1D<int> counter(1, 0);

    kernel_inclusive_scan<<<nb_blocks, blocksize, sizeof(int)>>>(predicate.data_, scan_A.data_, scan_P.data_, blockstates.data_, counter.data_, buffer_size);
    cudaDeviceSynchronize();

    CudaArray1D<int> shifted_predicate(buffer_size, 0);

    kernel_shift<<<nb_blocks, blocksize>>>(shifted_predicate.data_, predicate.data_, buffer_size);
    cudaDeviceSynchronize();

    predicate.free();

    CudaArray1D<int> image_data_copy(buffer_size);
    image_data_copy.copy_from(image_data.data_, cudaMemcpyDeviceToDevice);

    scatter_corresponding_adresses<<<nb_blocks, blocksize>>>(image_data.data_, image_data_copy.data_, shifted_predicate.data_, buffer_size);
    cudaDeviceSynchronize();

    shifted_predicate.free();

    apply_map_to_pixels<<<nb_blocks, blocksize>>>(image_data.data_, image_size);
    cudaDeviceSynchronize();

    // do histogram
    CudaArray1D<int> histogram(256, 0);

    // kernel_histogram<int><<<nb_blocks, blocksize>>>(image_data, histogram, image_size);
    kernel_histo<<<nb_blocks, blocksize, blocksize * sizeof(int)>>>(image_data.data_, histogram.data_, image_size);
    cudaDeviceSynchronize();

    // do inclusive scan
    cudaMemset(scan_A.data_, 0, nb_blocks * sizeof(int));
    cudaMemset(scan_P.data_, 0, nb_blocks * sizeof(int));
    cudaMemset(blockstates.data_, 0, nb_blocks * sizeof(int));
    cudaMemset(counter.data_, 0, sizeof(int));
    cudaCheckError();

    kernel_inclusive_scan<<<1, blocksize, sizeof(int)>>>(histogram.data_, scan_A.data_, scan_P.data_, blockstates.data_, counter.data_, 256);
    cudaDeviceSynchronize();

    // find first non zero
    CudaArray1D<int> first_non_zero(1, 0);

    CudaArray1D<int> predicate_find_first_non_zero(256, 0);

    kernel_filter_zeros<<<1, blocksize>>>(histogram.data_, predicate_find_first_non_zero.data_);
    cudaDeviceSynchronize();

    cudaMemset(scan_A.data_, 0, nb_blocks * sizeof(int));
    cudaMemset(scan_P.data_, 0, nb_blocks * sizeof(int));
    cudaMemset(blockstates.data_, 0, nb_blocks * sizeof(int));
    cudaMemset(counter.data_, 0, sizeof(int));

    kernel_inclusive_scan<<<1, blocksize, sizeof(int)>>>(predicate_find_first_non_zero.data_, scan_A.data_, scan_P.data_, blockstates.data_, counter.data_, 256);
    cudaDeviceSynchronize();

    scan_A.free();
    scan_P.free();
    blockstates.free();
    counter.free();

    kernel_find_first_non_zero<<<1, blocksize>>>(histogram.data_, predicate_find_first_non_zero.data_, first_non_zero.data_);
    cudaDeviceSynchronize();
    predicate_find_first_non_zero.free();

    // Apply map transformation of the histogram equalization
    image_data_copy.copy_from(image_data.data_, cudaMemcpyDeviceToDevice);

    kernel_apply_map_transformation<<<nb_blocks, blocksize>>>(image_data.data_, image_data_copy.data_, histogram.data_, first_non_zero.data_, image_size);
    cudaDeviceSynchronize();

    histogram.free();
    image_data_copy.free();
    first_non_zero.free();
}

void compute_statistics(std::vector<Image> &images)
{
#pragma omp parallel for
    for (int i = 0; i < (int)images.size(); ++i)
    {
        auto &image = images[i];
        const int image_size = image.width * image.height;

        CudaArray1D<int> d_reduce(1, 0);
        CudaArray1D<int> d_image(image_size);
        d_image.copy_from(image.buffer.data(), cudaMemcpyHostToDevice);

        constexpr int blocksize = 512;
        // Nb Blocks is not fixed : it should absolutely be modified and benchmarked to get the best performance
        const int nb_blocks = (image_size + blocksize - 1) / (blocksize);

        kernel_reduce<<<nb_blocks / 4, blocksize>>>(d_image.data_, d_reduce.data_, image_size);
        cudaDeviceSynchronize();

        d_reduce.copy_to((int *)&image.to_sort.total, cudaMemcpyDeviceToHost);

        d_image.free();
        d_reduce.free();
    }

    using ToSort = Image::ToSort;
    std::vector<ToSort> to_sort(images.size());
    std::generate(to_sort.begin(), to_sort.end(), [n = 0, images]() mutable
                  { return images[n++].to_sort; });
    std::sort(to_sort.begin(), to_sort.end(), [](ToSort a, ToSort b)
              { return a.total < b.total; });

    for (int i = 0; i < images.size(); ++i)
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
}
