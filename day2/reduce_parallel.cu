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
const int GRID_SIZE = 10240;

void timing(const real *d_x);
real reduce(const real *d_x);

__global__ void reduce_parallel(const real *d_x, real *d_y, const int length);

int main()
{
    real *h_x = (real *)malloc(M);
    for (int i = 0; i < N; ++i) {
        h_x[i] = 1.23;
    }
    real *d_x;
    CHECK(cudaMalloc(&d_x, M));
    CHECK(cudaMemcpy(d_x, h_x, M, cudaMemcpyHostToDevice));

    printf("\nUsing parallel:\n");
    timing(d_x);

    free(h_x);
    CHECK(cudaFree(d_x));
    return 0;
}

void timing(const real *d_x)
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

real reduce(const real *d_x)
{
    const int ymem = sizeof(real) * GRID_SIZE;
    const int smem = sizeof(real) * BLOCK_SIZE;

    real h_y[1] = {0};
    real *d_y;
    CHECK(cudaMalloc(&d_y, ymem));

    reduce_parallel<<<GRID_SIZE, BLOCK_SIZE, smem>>>(d_x, d_y, N);
    reduce_parallel<<<1, 1024, 1024*sizeof(real)>>>(d_y, d_y, GRID_SIZE);

    CHECK(cudaMemcpy(h_y, d_y, sizeof(real), cudaMemcpyDeviceToHost));
    CHECK(cudaFree(d_y));

    return h_y[0];
}

__global__ void reduce_parallel(const real *d_x, real *d_y, const int length)
{
    const int tid = threadIdx.x;
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;
    extern __shared__ real s_y[];

    real y = 0.0;
    for (int i = idx; i < length; i += stride) {
        y += d_x[i];
    }
    s_y[tid] = y;
    __syncthreads();

    for (int offset = blockDim.x >> 1; offset >= 32; offset >>= 1) {
        if (tid < offset) {
            s_y[tid] += s_y[tid + offset];
        }
        __syncthreads();
    }

    
    y = s_y[tid];
    thread_block_tile<32> g = tiled_partition<32>(this_thread_block());
    for (int offset = g.size() >> 1; offset > 0; offset >>= 1) {
        y += g.shfl_down(y, offset);
    }

    if (tid == 0) {
        d_y[blockIdx.x] = y;
    }
}
