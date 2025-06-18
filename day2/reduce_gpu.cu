#include "common.h"
#include <cuda_runtime.h>

#ifdef USE_DP
    typedef double real;
#else
    typedef float real;
#endif

const int NUM_REPEATS = 100;
const int N = 100000000;
const int M = sizeof(real) * N;
const int BLOCK_SIZE = 128;

void timing(real *h_x, real *d_x, const int method);
real reduce(real *d_x, const int method);

__global__ void reduce_global(real *d_x, real *d_y);
__global__ void reduce_shared(real *d_x, real *d_y);
__global__ void reduce_dynamic(real *d_x, real *d_y);

int main()
{
    real *h_x = (real *)malloc(M);
    for (int i = 0; i < N; ++i) {
        h_x[i] = 1.23;
    }
    real *d_x;
    CHECK(cudaMalloc(&d_x, M));

    printf("\nUsing global memory only:\n");
    timing(h_x, d_x, 0);

    
    // printf("\nUsing static shared memory:\n");
    // timing(h_x, d_x, 1);

    
    // printf("\nUsing dynamic shared memory:\n");
    // timing(h_x, d_x, 2);

    free(h_x);
    CHECK(cudaFree(d_x));
    return 0;
}

void timing(real *h_x, real *d_x, const int method)
{
    real sum = 0;

    for (int i = 0; i < NUM_REPEATS; ++i) {
        CHECK(cudaMemcpy(d_x, h_x, M, cudaMemcpyHostToDevice));

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

real reduce(real *d_x, const int method)
{
    int grid_size = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;

    const int ymem = sizeof(real) * grid_size;
    const int smem = sizeof(real) * BLOCK_SIZE;

    real *d_y;
    CHECK(cudaMalloc(&d_y, ymem));
    real *h_y = (real *)malloc(ymem);

    switch (method) {
        case 0:
            reduce_global<<<grid_size, BLOCK_SIZE>>>(d_x, d_y);
            break;
        case 1:
            reduce_shared<<<grid_size, BLOCK_SIZE>>>(d_x, d_y);
            break;
        case 2:
            reduce_dynamic<<<grid_size, BLOCK_SIZE, smem>>>(d_x, d_y);
            break;
        default:
            printf("Error: wrong method\n");
            exit(1);
    }

    CHECK(cudaMemcpy(h_y, d_y, ymem, cudaMemcpyDeviceToHost));

    real result = 0.0;
    for (int i = 0; i < grid_size; ++i) {
        result += h_y[i];
    }

    free(h_y);
    CHECK(cudaFree(d_y));
    return result;
}

__global__ void reduce_global(real *d_x, real *d_y)
{
    const int tid = threadIdx.x;
    real *block_x = d_x + blockIdx.x * blockDim.x;

    for (int offset = blockDim.x >> 1; offset > 0; offset >>= 1) {
        if (tid < offset) {
            block_x[tid] += block_x[tid + offset];
        }
        __syncthreads();
    }

    if (tid == 0) {
        d_y[blockIdx.x] = block_x[0];
    }
    return;
}

__global__ void reduce_shared(real *d_x, real *d_y)
{

}

__global__ void reduce_dynamic(real *d_x, real *d_y)
{

}