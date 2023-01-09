#include <iostream>

#include <chrono>

#include "fix_cpu.hh"
#include "fix_gpu.cuh"
#include "cub_version.cuh"

int main([[maybe_unused]] int argc, [[maybe_unused]] char *argv[])
{
    // -- Pipeline initialization

    std::cout << "File loading..." << std::endl;

    // - Get file paths

    using recursive_directory_iterator = std::filesystem::recursive_directory_iterator;
    std::vector<std::string> filepaths;
    for (const auto &dir_entry : recursive_directory_iterator("../images"))
        filepaths.emplace_back(dir_entry.path().string());

    // - Init pipeline object

    Pipeline pipeline_cpu(filepaths);
    Pipeline pipeline_gpu(filepaths);
    Pipeline pipeline_cub(filepaths);

    CudaArray1D<int> init(1);
    init.free();

    auto t1 = std::chrono::system_clock::now();

    std::cout << "cpu main" << std::endl;
    cpu_main(argc, argv, pipeline_cpu);

    auto t2 = std::chrono::system_clock::now();

    std::cout << "gpu main" << std::endl;
    gpu_main(argc, argv, pipeline_gpu);

    auto t3 = std::chrono::system_clock::now();

    std::cout << "cub main" << std::endl;
    cub_main(argc, argv, pipeline_cub);

    auto t4 = std::chrono::system_clock::now();

    std::chrono::duration<double> cpu_time = t2 - t1;
    std::chrono::duration<double> gpu_time = t3 - t2;
    std::chrono::duration<double> cub_time = t4 - t3;

    std::cout << "CPU time: " << cpu_time.count() << "\n";
    std::cout << "GPU time: " << gpu_time.count() << "\n";
    std::cout << "GPU with CUB: " << cub_time.count() << "\n";

    return 0;
}