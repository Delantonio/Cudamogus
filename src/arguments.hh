#include <string>

enum compute_version
{
    CPU,
    GPU,
    GPUandCUB,
};

struct Arguments
{
    bool benchmark;
    compute_version version;
};
