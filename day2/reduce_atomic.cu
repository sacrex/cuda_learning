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

void timing(real *d_x);
real reduce(real *d_x);

__global__ void reduce_atomic(real *d_x, real *d_y);

int main()
{
    real *h_x = (real *)malloc(M);
    for (int i = 0; i < N; ++i) {
        h_x[i] = 1.23;
    }
    real *d_x;
    CHECK(cudaMalloc(&d_x, M));
    CHECK(cudaMemcpy(d_x, h_x, M, cudaMemcpyHostToDevice));

    printf("\nUsing atomicAdd:\n");
    timing(d_x);

    free(h_x);
    CHECK(cudaFree(d_x));
    return 0;
}

void timing(real *d_x)
{
    real sum = 0;

    for (int i = 0; i < NUM_REPEATS; ++i) {
        cudaEvent_t start, end;
        CHECK(cudaEventCreate(&start));
        CHECK(cudaEventCreate(&end));
        CHECK(cudaEventRecord(start));
        cudaEventQuery(start);

        sum = reduce(d_x);

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

real reduce(real *d_x)
{
    int grid_size = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    const int smem = sizeof(real) * BLOCK_SIZE;

    real h_y[1] = {0};
    real *d_y;
    CHECK(cudaMalloc(&d_y, sizeof(real)));
    CHECK(cudaMemcpy(d_y, h_y, sizeof(real), cudaMemcpyHostToDevice));

    reduce_atomic<<<grid_size, BLOCK_SIZE, smem>>>(d_x, d_y);

    CHECK(cudaMemcpy(h_y, d_y, sizeof(real), cudaMemcpyDeviceToHost));

    CHECK(cudaFree(d_y));
    return h_y[0];
}

__global__ void reduce_atomic(real *d_x, real *d_y)
{
    const int tid = threadIdx.x;
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    extern __shared__ real s_y[];

    s_y[tid] = (idx < N) ? d_x[idx]: 0.0;
    __syncthreads();

    for (int offset = blockDim.x >> 1; offset > 0; offset >>= 1) {
        if (tid < offset) {
            s_y[tid] += s_y[tid + offset];
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicAdd(d_y, s_y[0]);
    }
}