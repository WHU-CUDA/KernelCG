 //operator 1
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>

#ifndef _UNISTD_H
#define _UNISTD_H
#include <io.h>
//#include <process.h>
#endif /* _UNISTD_H */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include "time.h"
//#include <unistd.h>
//#include <sys/time.h>
#include <cuda.h>
#include<Windows.h>
#include "polybenchUtilFuncts.h"
#include <cuda_runtime.h>
#include<device_launch_parameters.h>
#include "kernel_cg.h"
#define GPU_DEVICE 0

float list[2][N];
//define the error threshold for the results "not matching"
#define PERCENT_DIFF_ERROR_THRESHOLD 0.05

/* Thread block dimensions */
#define DIM_THREAD_BLOCK_X 32
#define DIM_THREAD_BLOCK_Y 8

/*operator 1*/
typedef int oa_int3[3];
#define min(a,b) ((a)<(b))?(a):(b)
#define BLOCK_NUM 32
#define calc_id2(i,j,k,S0,S1) ((k)*(S0)*(S1)+(j)*(S0)+(i))
#define N 32
/* Can switch DATA_TYPE between DATA_TYPE and double */
typedef float DATA_TYPE;
// int Pn = 100;  //执行数目
// DATA_TYPE CPU1 = 0.014904;//一个cpu任务时间
// DATA_TYPE aa = 0.00109;//一个gpu任务时间
//cout << “cpu数目”<<(Pn* aa) / (3 *  aa + CPU1) << endl;
// int CPUn = int(((Pn * aa) / (threadnum * aa + CPU1)));  //cpu数目
// int GPUn = Pn - threadnum * CPUn;  //gpu数目
/**
 * 初始化GPU参数
 * @prop GPU_DEVICE gpu的设备号，这里设定为0
 * */
void GPU_argv_init()
{
	cudaDeviceProp deviceProp;
	cudaGetDeviceProperties(&deviceProp, GPU_DEVICE);
	//printf("setting device %d with name %s\n",GPU_DEVICE,deviceProp.name);
	cudaSetDevice(GPU_DEVICE);
}

extern "C" __declspec(dllexport) void init(int cpu, int gpu, int op, int threadNum, int left[3], int right[3], float factor, float **list) {
	CUDA_CONF::CPUn = cpu;
	CUDA_CONF::GPUn = gpu;
	CUDA_CONF::threadnum = threadNum;
	CALCULATE_CONF::factor = factor;
	CALCULATE_CONF::op = op;
	CALCULATE_CONF::left = left;
	CALCULATE_CONF::right = right;
	CALCULATE_DATA::list = list;
	printf("CPUn is %d, GPUn is %d, OP is %d, threadNum is %d, Factor is %f\n", cpu, gpu, op, threadNum, factor);
}


/**
 * 算子在CPU上运行程序
 * list 为需要计算的数据
 * o 未知数据，也许是定位符
 * left 左边算式的参数[1, 1, 1] 则为calc_id2(1+i, 1+j, 1+k, s0_0, s0_1)
 * right 同上为右边算式的参数
 * factor为左边算式的系数，这里可以为1.0或0.5
 * op 为规定左右两个算式的加减 1 为 + , -1 为 -
 * */
extern "C" __declspec(dllexport) void kernel_cpu(float list[][N], int o, int left[], int right[], float factor, int op) {
  //o = 1;//temp wangdong
  oa_int3* oa_int3_p = (oa_int3*)(list[2]);
  float *list_0;  list_0 = (float *) list[0];
  float *list_1;  list_1 = (float *) list[1];

  const oa_int3 &lbound = oa_int3_p[2];
  const oa_int3 &rbound = oa_int3_p[3];
  const oa_int3 &sp = oa_int3_p[4];

  const int S0_0 = oa_int3_p[0][0];    const int S0_1 = oa_int3_p[0][1];
  const int S1_0 = oa_int3_p[1][0];    const int S1_1 = oa_int3_p[1][1];
  int ist=o ;   int ied=o + sp[0] ;
  int jst=o ;   int jed=o + sp[1] ;
  int kst=o ;   int ked=o + sp[2] ;

  /*for (int kk = kst; kk< ked+BLOCK_NUM; kk += BLOCK_NUM)*/{
    //int kend=min(kk+BLOCK_NUM,ked);
    /*for (int jj = jst; jj< jed+BLOCK_NUM; jj += BLOCK_NUM)*/{
      //int jend=min(jj+BLOCK_NUM,jed);
      /*for (int ii = ist; ii< ied+BLOCK_NUM; ii += BLOCK_NUM)*/{
        //int iend=min(ii+BLOCK_NUM,ied);
        for (int k = kst; k < ked; k++) {
          for (int j = jst; j < jed; j++) {
            #pragma simd
            #pragma clang loop vectorize(assume_safety)
            #pragma clang loop interleave(enable)
            #pragma clang loop vectorize_width(8) interleave_count(1)
            for (int i = ist; i < ied ;i++){
              list_1[calc_id2(i,j,k,S1_0,S1_1)] = factor*((list_0[calc_id2(i+left[0],j+left[1],k+left[2],S0_0,S0_1)])+(op)*(list_0[calc_id2(i+right[0],j+right[1],k+right[2],S0_0,S0_1)]));
            }
          }
        }
      }
    }
  }
  return ;
}




__global__ void kernel_gpu(DATA_TYPE* list_0, DATA_TYPE* list_1, int ied, int jed, int ked, int S0_0, int S0_1, int S1_0, int S1_1, int o, int op, int* left, int * right, float factor) {
	int i = threadIdx.x + blockIdx.x * blockDim.x + o;
	int j = threadIdx.y + blockIdx.y * blockDim.y + o;
	int k = threadIdx.z + blockIdx.z * blockDim.z + o;

	if (i >= ied || j >= jed || k >= ked)  return;
	else {

		list_1[calc_id2(i, j, k, S1_0, S1_1)] = factor * ((list_0[calc_id2(left[0]+i, left[1]+j, left[2]+o, S0_0, S0_1)]) + (op)*(list_0[calc_id2(right[0]+i, right[1]+j, right[2]+o, S0_0, S0_1)]));
	}
	//__syncthreads();
}

extern "C" __declspec(dllexport) void kernel(DATA_TYPE list[][N], int o) {
		//o = 1;//temp wangdong
		oa_int3* oa_int3_p = (oa_int3*)(list[2]);

		DATA_TYPE* list_0;  list_0 = (DATA_TYPE*)list[0];
		DATA_TYPE* list_1;  list_1 = (DATA_TYPE*)list[1];

		const oa_int3& lbound = oa_int3_p[2];
		const oa_int3& rbound = oa_int3_p[3];
		const oa_int3& sp = oa_int3_p[4];

		const int S0_0 = oa_int3_p[0][0];    const int S0_1 = oa_int3_p[0][1];
		const int S1_0 = oa_int3_p[1][0];    const int S1_1 = oa_int3_p[1][1];
		int ist = o;   int ied = o + sp[0];
		int jst = o;   int jed = o + sp[1];
		int kst = o;   int ked = o + sp[2];

		int edge_x = 32, edge_y = 32, edge_z = 1;
		dim3 threadPerBlock(edge_x, edge_y, edge_z);
		dim3 num_blocks((sp[0] + edge_x - 1) / edge_x, (sp[1] + edge_y - 1) / edge_y, (sp[2] + edge_z - 1) / edge_z);
		kernel_gpu <<<num_blocks, threadPerBlock >>> (list_0, list_1, ied, jed, ked, S0_0, S0_1, S1_0, S1_1, o, CALCULATE_CONF::op, CALCULATE_CONF::left, CALCULATE_CONF::right, CALCULATE_CONF::factor);
		return;
}

DWORD WINAPI fun(LPVOID p)
{
	clock_t begin, end;
	begin = clock();
	DATA_TYPE list[2][N]; int o = 1;
	for (int i = 0; i < CUDA_CONF::CPUn; i++)
	{
		kernel_cpu(list, o, CALCULATE_CONF::left, CALCULATE_CONF::right, CALCULATE_CONF::factor, CALCULATE_CONF::op);
	}
	end = clock();

	printf("cpubegin :%d \n", begin);
	printf("cpuend :%d \n", end);
	double time = (double)(end - begin) / CLOCKS_PER_SEC;
	printf("cpu cost %lf s.\n", time);
	return 0;
}

extern "C" __declspec(dllexport) void run(DATA_TYPE list[2][N]) {
	double t_start, t_end;
	clock_t begin, end;
	begin = clock();
	int o = 1;
	DATA_TYPE* list_0;
	DATA_TYPE* list_1;
	int ied; int jed;
	int ked; int S0_0;
	int S0_1; int S1_0; int S1_1;
	GPU_argv_init();
	oa_int3* oa_int3_p = (oa_int3*)(list[2]);

	const oa_int3& lbound = oa_int3_p[2];
	const oa_int3& rbound = oa_int3_p[3];
	const oa_int3& sp = oa_int3_p[4];

	for(int i = 0; i < CUDA_CONF::threadnum; i++) {
		CreateThread(NULL, 0,	fun, 0, 0, NULL); // 创建线程
	}

	for (int i = 0; i < CUDA_CONF::GPUn; i++)
	{
		kernel(list,o);
	}
	end = clock();
	printf("cg begin :%d \n", begin);
	printf("cg end :%d \n", end);
	double time = (double)(end - begin) / CLOCKS_PER_SEC;
	printf("Time of  cg running :  %lf s\n", time);
}


// int main(int argc, char *argv[]) {
// 	// run();
// 	return 0;
// }
