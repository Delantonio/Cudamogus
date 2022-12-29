#include "image.hh"
#include "pipeline.hh"
#include "fix_cpu.hh"

#include <vector>
#include <iostream>
#include <algorithm>
#include <sstream>
#include <filesystem>
#include <numeric>

__global__ void reduce1(int *g_idata, int *g_odata, int size)
{
    extern __shared__ int sdata[];

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= size)
        return;
    sdata[tid] = g_idata[i];
    __syncthreads();

    for (unsigned int s = 1; s < blockDim.x; s *= 2)
    {
        int index = 2 * s * tid;
        if (index < blockDim.x)
        {
            sdata[index] += sdata[index + s];
        }
        __syncthreads();
    }

    if (tid == 0)
    {
        g_odata[blockIdx.x] = sdata[0];
    }
}

int main([[maybe_unused]] int argc, [[maybe_unused]] char* argv[])
{
    // -- Pipeline initialization

    std::cout << "File loading..." << std::endl;

    // - Get file paths

    using recursive_directory_iterator = std::filesystem::recursive_directory_iterator;
    std::vector<std::string> filepaths;
    for (const auto& dir_entry : recursive_directory_iterator("../images"))
        filepaths.emplace_back(dir_entry.path());

    // - Init pipeline object

    Pipeline pipeline(filepaths);

    // -- Main loop containing image retring from pipeline and fixing

    const int nb_images = pipeline.images.size();
    std::vector<Image> images(nb_images);

    // - One CPU thread is launched for each image

    std::cout << "Done, starting compute" << std::endl;

    #pragma omp parallel for
    for (int i = 0; i < nb_images; ++i)
    {
        // TODO : make it GPU compatible (aka faster)
        // You will need to copy images one by one on the GPU
        // You can store the images the way you want on the GPU
        // But you should treat the pipeline as a pipeline :
        // You *must not* copy all the images and only then do the computations
        // You must get the image from the pipeline as they arrive and launch computations right away
        // There are still ways to speeds this process of course (wait for last class)
        images[i] = pipeline.get_image(i);
        fix_image_cpu(images[i]);
    }

    std::cout << "Done with compute, starting stats" << std::endl;

    // -- All images are now fixed : compute stats (total then sort)

    // - First compute the total of each image

    // TODO : make it GPU compatible (aka faster)
    // You can use multiple CPU threads for your GPU version using openmp or not
    // Up to you :)

    // #pragma omp parallel for
    // for (int i = 0; i < nb_images; ++i)
    // {
    //     auto& image = images[i];
    //     const int image_size = image.width * image.height;
    //     image.to_sort.total = std::reduce(image.buffer.cbegin(), image.buffer.cbegin() + image_size, 0);
    // }
    
    const int block_size = 1024;
    const int grid_size = (nb_images + block_size - 1) / block_size;
    dim3 nb_threads(32, 32);

    #pragma omp parallel for
    for (int i = 0; i < nb_images; i++)
    {
        dim3 nb_blocks((images[i].width + nb_threads.x - 1) / nb_threads.x,
                       (images[i].height + nb_threads.y - 1) / nb_threads.y);
        
        int buffer_size = images[i].width * images[i].height;

        int *in_buffer;
        int *out_buffer;
        int *tmp;

        cudaMalloc((void**)&in_buffer, buffer_size * sizeof(int));
        cudaMalloc((void**)&out_buffer, buffer_size * sizeof(int));

        cudaMemcpy(&in_buffer, images[i].buffer.data(), buffer_size * sizeof(int), cudaMemcpyHostToDevice);

        // Local reduction
        reduce1<<<nb_blocks, nb_threads, sizeof(int) * block_size>>>(in_buffer, tmp, buffer_size);
        // Global reduction
        reduce1<<<block_size, 1, sizeof(int) * block_size>>>(tmp, out_buffer, buffer_size);

        cudaMemcpy(&images[i].to_sort.total, out_buffer, sizeof(int), cudaMemcpyDeviceToHost);
        
        printf("Total : %d", images[i].to_sort.total);
        //printf("Total : %d", out_buffer[0]);

        cudaFree(in_buffer);
        cudaFree(out_buffer);
    }


    // - All totals are known, sort images accordingly (OPTIONAL)
    // Moving the actual images is too expensive, sort image indices instead
    // Copying to an id array and sort it instead

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
    // If you did the sorting, check that the ids are in the same order
    for (int i = 0; i < nb_images; ++i)
    {
        std::cout << "Image #" << images[i].to_sort.id << " total : " << images[i].to_sort.total << std::endl;
        std::ostringstream oss;
        oss << "Image#" << images[i].to_sort.id << ".pgm";
        std::string str = oss.str();
        images[i].write(str);
    }

    std::cout << "Done, the internet is safe now :)" << std::endl;

    return 0;
}
