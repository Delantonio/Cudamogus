#include <iostream>

#include <chrono>

#include "fix_cpu.hh"
#include "fix_gpu.cuh"
#include "cub_version.cuh"

void compare_results(const Pipeline &ref, const Pipeline &other)
{
    bool no_problem = true;
    for (size_t i = 0; i < ref.images.size(); i++)
    {
        if (ref.images[i].to_sort.total != other.images[i].to_sort.total)
        {
            std::cout << "Error detected at image " << ref.images[i].to_sort.id << ": ref = "
                      << ref.images[i].to_sort.total << ", other = " << other.images[i].to_sort.total << "\n";
            no_problem = false;
        }
    }

    if (no_problem)
        std::cout << "Results are valid!\n";
}

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

    // Load the CUDA context ahead of time
    CudaArray1D<int> init(1);
    init.free();

    auto t1 = std::chrono::system_clock::now();

    std::cout << "cpu main" << std::endl;
    cpu_main(argc, argv, pipeline_cpu, false);

    auto t2 = std::chrono::system_clock::now();

    std::cout << "gpu main" << std::endl;
    gpu_main(argc, argv, pipeline_gpu, false);

    auto t3 = std::chrono::system_clock::now();

    std::cout << "cub main" << std::endl;
    cub_main(argc, argv, pipeline_cub, false);

    auto t4 = std::chrono::system_clock::now();

    std::chrono::duration<double> cpu_time = t2 - t1;
    std::chrono::duration<double> gpu_time = t3 - t2;
    std::chrono::duration<double> cub_time = t4 - t3;

    std::cout << "\n\n";

    std::cout << "CPU time: " << cpu_time.count() << "\n";
    std::cout << "GPU time: " << gpu_time.count() << "\n";
    std::cout << "GPU with CUB: " << cub_time.count() << "\n\n";

    std::cout << "Checking GPU results...\n";
    compare_results(pipeline_cpu, pipeline_gpu);

    std::cout << "\nChecking GPU + CUB results...\n";
    compare_results(pipeline_cpu, pipeline_cub);

    return 0;
}