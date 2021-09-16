
#ifndef KERNEL_CG
#define KERNEL_CG
// #include <string>
#include "argtable3.h"

// Linux下
#include <pthread.h>


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
    int left[3] = {0,0,0}; // 默认为i,j,k
    int right[3] = {0,0,0}; // 默认为i,j,k
    float factor = 0.5;
}

namespace CACULATE_DATA 
{
    int coloum = 0;
    int line = 0;
}

struct arg_int *gpu, *cpu, *opreator, *tn, *l, *r,*coloum, *line;
struct arg_dbl *fc;
struct arg_lit *verb, *help, *version;
struct arg_end *end;
#define VERSION 0x020101

#endif
