#include <iostream>

#include "fix_cpu.hh"
#include "fix_gpu.cuh"

int main([[maybe_unused]] int argc, [[maybe_unused]] char *argv[])
{
    {
        // std::cout << "cub version : " << std::endl;
        // cub_main(argc, argv);
    }

    // std::cout << "cpu main" << std::endl;
    // cpu_main(argc, argv);

    std::cout << "gpu main" << std::endl;
    gpu_main(argc, argv);

    // if (argc > 1)
    //     cpu_main(argc, argv);
    // else
    //     gpu_main(argc, argv);

    return 0;
}