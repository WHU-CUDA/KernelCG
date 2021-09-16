
#ifndef KERNEL_CG
#define KERNEL_CG
#include <string>

#define N 32
namespace CUDA_CONF
{
    int CPUn = 0; // CPU的使用数量
    int GPUn = 0; // GPU的使用数量
    int threadnum = 1; // 线程数目
};

namespace CALCULATE_CONF
{
    int op = 1; // 默认op是+
    int *left; // 默认为i,j,k
    int *right; // 默认为i,j,k
    float factor = 0.5;
}

namespace CALCULATE_DATA 
{
    float **list;
}

#define VERSION 0x020101

#endif
