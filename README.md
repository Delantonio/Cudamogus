# IRGPUA
## AUTHORS

Antoine Delattre, Arthur Le Bourg & Guillaume Poisson

### Requirements to build

* [Cuda Toolkit](https://developer.nvidia.com/cuda-downloads)
* C++ compiler ([g++](https://gcc.gnu.org/) for linux,  [MSVC](https://visualstudio.microsoft.com/downloads/) for Windows)
* [GPU supported by CUDA](https://en.wikipedia.org/wiki/CUDA#GPUs_supported)
* [CMake](https://cmake.org/download/)
* [CUB](https://docs.nvidia.com/cuda/cub/index.html)

### Build

- To build, execute the following commands :

```bash
mkdir build && cd build
cmake ..
make -j
```

## Program Options

- Use `--benchmark` to run a comparison between the cpu, gpu and gpu with libraries versions. Also checks if the results are correct.
- Use `--version <cpu|gpu|gpucub>` to run a specific version of the program on the images. The images are saved at the end of the program.