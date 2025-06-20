#include "common.h"
#include <cuda_runtime.h>
#include <cooperative_groups.h>
using namespace cooperative_groups;


#ifdef USE_DP
    typedef double real;
#else
    typedef float real;
#endif

const int NUM_REPEATS = 100;
const int N = 100000000;
const int M = sizeof(real) * N;
const int BLOCK_SIZE = 128;
const unsigned FULL_MASK = 0xffffffff;

void timing(const real *d_x, const int method);
real reduce(const real *d_x, const int method);

__global__ void reduce_syncwarp(const real *d_x, real *d_y);
__global__ void reduce_shfl(const real *d_x, real *d_y);
__global__ void reduce_cp(const real *d_x, real *d_y);

int main()
{
    real *h_x = (real *)malloc(M);
    for (int i = 0; i < N; ++i) {
        h_x[i] = 1.23;
    }
    real *d_x;
    CHECK(cudaMalloc(&d_x, M));
    CHECK(cudaMemcpy(d_x, h_x, M, cudaMemcpyHostToDevice));

    printf("\nUsing syncwarp:\n");
    timing(d_x, 0);

    
    printf("\nUsing shfl:\n");
    timing(d_x, 1);

    
    printf("\nUsing cooperative group:\n");
    timing(d_x, 2);

    free(h_x);
    CHECK(cudaFree(d_x));
    return 0;
}

void timing(const real *d_x, const int method)
{
    real sum = 0;

    for (int i = 0; i < NUM_REPEATS; ++i) {
        cudaEvent_t start, end;
        CHECK(cudaEventCreate(&start));
        CHECK(cudaEventCreate(&end));
        CHECK(cudaEventRecord(start));
        cudaEventQuery(start);

        sum = reduce(d_x, method);

        CHECK(cudaEventRecord(end));
        CHECK(cudaEventSynchronize(end));

        float elpased_time;
        CHECK(cudaEventElapsedTime(&elpased_time, start, end));
        // printf("i = %d, Time = %g ms.\n", i+1, elpased_time);

        CHECK(cudaEventDestroy(start));
        CHECK(cudaEventDestroy(end));
    }

    printf("sum = %f.\n", sum);
}

real reduce(const real *d_x, const int method)
{
    int grid_size = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    const int smem = sizeof(real) * BLOCK_SIZE;

    real h_y[1] = {0};
    real *d_y;
    CHECK(cudaMalloc(&d_y, sizeof(real)));
    CHECK(cudaMemcpy(d_y, h_y, sizeof(real), cudaMemcpyHostToDevice));

    switch (method) {
        case 0:
            reduce_syncwarp<<<grid_size, BLOCK_SIZE, smem>>>(d_x, d_y);
            break;
        case 1:
            reduce_shfl<<<grid_size, BLOCK_SIZE, smem>>>(d_x, d_y);
            break;
        case 2:
            reduce_cp<<<grid_size, BLOCK_SIZE, smem>>>(d_x, d_y);
            break;
        default:
            printf("Error: wrong method\n");
            exit(1);
    }

    CHECK(cudaMemcpy(h_y, d_y, sizeof(real), cudaMemcpyDeviceToHost));
    CHECK(cudaFree(d_y));

    return h_y[0];
}

__global__ void reduce_syncwarp(const real *d_x, real *d_y)
{
    const int tid = threadIdx.x;
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    extern __shared__ real s_y[];

    s_y[tid] = (idx < N) ? d_x[idx]: 0.0;
    __syncthreads();

    for (int offset = blockDim.x >> 1; offset >= 32; offset >>= 1) {
        if (tid < offset) {
            s_y[tid] += s_y[tid + offset];
        }
        __syncthreads();
    }

    for (int offset = 16; offset > 0; offset >>= 1) {
        if (tid < offset) {
            s_y[tid] += s_y[tid + offset];
        }
        __syncwarp();
    }

    if (tid == 0) {
        atomicAdd(d_y, s_y[0]);
    }
}

__global__ void reduce_shfl(const real *d_x, real *d_y)
{
    const int tid = threadIdx.x;
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    extern __shared__ real s_y[];

    s_y[tid] = (idx < N) ? d_x[idx]: 0.0;
    __syncthreads();

    for (int offset = blockDim.x >> 1; offset >= 32; offset >>= 1) {
        if (tid < offset) {
            s_y[tid] += s_y[tid + offset];
        }
        __syncthreads();
    }

    real y = s_y[tid];
    
    for (int offset = 16; offset > 0; offset >>= 1) {
        y += __shfl_down_sync(FULL_MASK, y, offset);
    }

    if (tid == 0) {
        atomicAdd(d_y, y);
    }
}

__global__ void reduce_cp(const real *d_x, real *d_y)
{
    const int tid = threadIdx.x;
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    extern __shared__ real s_y[];

    s_y[tid] = (idx < N) ? d_x[idx]: 0.0;
    __syncthreads();

    for (int offset = blockDim.x >> 1; offset >= 32; offset >>= 1) {
        if (tid < offset) {
            s_y[tid] += s_y[tid + offset];
        }
        __syncthreads();
    }

    real y = s_y[tid];
    thread_block_tile<32> g = tiled_partition<32>(this_thread_block());
    for (int i = g.size() >> 1; i > 0; i >>= 1) {
        y += g.shfl_down(y, i);
    }

    if (tid == 0) {
        atomicAdd(d_y, y);
    }
}
