#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <cuda_runtime.h>

#define CHECK(ret)                                                      \
do                                                                      \
{                                                                       \
    const cudaError_t error_code = ret;                                 \
    if (error_code != cudaSuccess) {                                    \
        printf("CUDA Error: \n");                                       \
        printf("    File: %s\n", __FILE__);                             \
        printf("    LINE: %d\n", __LINE__);                             \
        printf("    Error Code: %d\n", error_code);                     \
        printf("    Error Msg: %s\n", cudaGetErrorString(error_code));  \
        exit(1);                                                        \
    }                                                                   \
} while(0)

#define OFFSET(row, col, ld) ((row) * (ld) + (col))
#define FLOAT4(val) (reinterpret_cast<float4*>(&(val))[0])

typedef void (*Sgemm) (float *, float *, float *, const int, const int, const int);

float testError(Sgemm gpuSgemm, dim3 gridDim, dim3 blockDim, const int, const int, const int);
float testPerformance(Sgemm gpuSgemm, dim3 gridDim, dim3 blockDim, const int, const int, const int, const int);

void cpuSgemm(float *a, float *b, float *c, const int M, const int N, const int K) {
    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {
            float psum = 0.0;
            for (int k = 0; k < K; ++k) {
                psum += a[OFFSET(m, k, K)] * b[OFFSET(k, n, N)];
            }
            c[OFFSET(m, n, N)] = psum;
        }
    }
}


// block-tiling + thread-tiling + share_memory + solve bank conflict
__global__ void sgemm_v2(float * __restrict__ a, float * __restrict__ b,
                float * __restrict__ c, const int M, const int N, const int K) {
    const int BM = 128, BN = 128, BK = 8;  // block tiling
    const int TM = 8, TN = 8;  // thread tiling

    const int bx = blockIdx.x;
    const int by = blockIdx.y;
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int tid = ty * blockDim.x + tx;

    //当计算C中的BM*BN矩阵时, 每次从A中一个读取BM*BK的块, 共读取K/BK次,并且一个block中共有(BM/TM * BN/TN)个线程,故
    //可以直接计算出每个线程读取 BM*BK / (BM/TM * BN/TN) = (128*8) / (128/8 * 128/8) = 4个元素。
    //与v1版不同，这里s_a是[BK][BM],转置模式
    __shared__ float s_a[BK][BM];
    __shared__ float s_b[BK][BN];

    //用来临时存储global memory -> share memory的转化
    float r_load_a[4];
    float r_load_b[4];

    //读取share memory到寄存器
    float r_comp_a[TM], r_comp_b[TN];
    float r_c[TM][TN] = {0.0};

    
    // 计算每个线程在share memory中行列索引 (s_a)
    int load_a_smem_m = tid >> 1; //因为每个线程读取4个元素，每行为BK大小，共8个元素，故1行需要2个线程，那么 tid >> 1 可以计算该线程对应哪一行.
    int load_a_smem_k = (tid & 1) << 2; //计算列值, 例如1号线程处理第一行的4-7, 3号处理第二行的4-7. 也就是偶数线程处理每行开始的4个元素,而奇数线程计算每行第4个开始的4个元素.
    // 计算每个线程在share memory中行列索引 (s_b)
    int load_b_smem_k = tid >> 5; //因为读取的[BK,BN],每行一共128个元素,同样每个线程处理4个,那么一行需要32个线程, 那么tid >> 5可以计算该线程对应哪一行.
    int load_b_smem_n = (tid % 32) * 4; // (tid & 31) << 2; 计算该线程处理哪一列,因为一行需要32个线程处理,那么 (tid % 32) 或则 (tid & 31)都计算处理列号,然后 *4 即可.(由于每个线程处理4个元素)

    int load_a_gmem_m = by * BM + load_a_smem_m;
    int load_b_gmem_n = bx * BN + load_b_smem_n;

    // block tiling
    for (int bk = 0; bk < (K + BK - 1) / BK; ++bk) {
        //从A矩阵读取[BM,BK]到共享内存,每个线程读取4个元素.
        int load_a_gmem_k = bk * BK + load_a_smem_k; // 从矩阵A中读取的列
        int load_a_gmem_addr = OFFSET(load_a_gmem_m, load_a_gmem_k, K); // load_a_gmem_m * K + load_a_gmem_k: 计算从A矩阵读取的位置
        FLOAT4(r_load_a[0]) = FLOAT4(a[load_a_gmem_addr]);

        s_a[load_a_smem_k    ][load_a_smem_m] = r_load_a[0];
        s_a[load_a_smem_k + 1][load_a_smem_m] = r_load_a[1];
        s_a[load_a_smem_k + 2][load_a_smem_m] = r_load_a[2];
        s_a[load_a_smem_k + 3][load_a_smem_m] = r_load_a[3];

        //从B矩阵读取[BK,BN]到共享内存,每个线程读取4个元素.
        int load_b_gmem_k = bk * BK + load_b_smem_k;  // 从矩阵B中读取的行
        int load_b_gmem_addr = OFFSET(load_b_gmem_k, load_b_gmem_n, N);

        FLOAT4(r_load_b[0]) = FLOAT4(b[load_b_gmem_addr]);
        FLOAT4(s_b[load_b_smem_k][load_b_smem_n]) = FLOAT4(r_load_b[0]);


        __syncthreads(); //让每个线程执行完,那么这样s_a和s_b就已经存好了A中的一个block tile(BM,BK)和B中的一个tile(BK,BN).

        // thread tiling
        // 每个线程从s_a和s_b中取数据,计算自己的TM*TN的矩阵
        #pragma unroll
        for (int k = 0; k < BK; ++k) {
            //每个线程读取从s_a和s_b分别读取8个元素，为了解决bank conflict,读取的2个4元素，相隔一半读取。
            FLOAT4(r_comp_a[0]) = FLOAT4(s_a[k][ty * TM / 2]);
            FLOAT4(r_comp_a[4]) = FLOAT4(s_a[k][ty * TM / 2 + BM / 2]);
            FLOAT4(r_comp_b[0]) = FLOAT4(s_b[k][tx * TN / 2]);
            FLOAT4(r_comp_b[4]) = FLOAT4(s_b[k][tx * TN / 2 + BN / 2]);
        
            #pragma unroll
            for (int m = 0; m < TM; ++m) {
                #pragma unroll
                for (int n = 0; n < TN; ++n) {
                    r_c[m][n] += r_comp_a[m] * r_comp_b[n];
                }
            }
            __syncthreads();
        }
    }

    // 把r_c存到全局内存中,按上下左右4块存储
    #pragma unroll
    for (int i = 0; i < TM / 2; ++i) {
        int c_gemm_m = by * BM + ty * TM / 2 + i;
        int c_gemm_n = bx * BN + tx * TN / 2;
        int c_gemm_addr = OFFSET(c_gemm_m, c_gemm_n, N);
        // 越界检查！！
        if (c_gemm_m < M && c_gemm_n < N) {
            FLOAT4(c[c_gemm_addr]) = FLOAT4(r_c[i][0]);
        }
        if (c_gemm_m < M && c_gemm_n < N - BN / 2) {
            FLOAT4(c[c_gemm_addr + BN / 2]) = FLOAT4(r_c[i][4]);
        }
    }
    #pragma unroll
    for (int i = 0; i < TM / 2; ++i) {
        int c_gemm_m = by * BM + BM / 2 + ty * TM / 2 + i;
        int c_gemm_n = bx * BN + tx * TN / 2;
        int c_gemm_addr = OFFSET(c_gemm_m, c_gemm_n, N);
        if (c_gemm_m < M && c_gemm_n < N) {
            FLOAT4(c[c_gemm_addr]) = FLOAT4(r_c[i + TM / 2][0]);
        }
        if (c_gemm_m < M && c_gemm_n < N - BN / 2) {
            FLOAT4(c[c_gemm_addr + BN / 2]) = FLOAT4(r_c[i + TM / 2][4]);
        }
    }
}

int main() {
    const int outer_repeat = 10, inner_repeat = 1;
    const int BM = 128, BN = 128, TM = 8, TN = 8;

    {
        const int M = 320, N = 320, K = 8;
        dim3 blockDim(BN / TN, BM / TM);
        dim3 gridDim((N + BN - 1) / BN, (M + BM - 1) / BM);
        float max_error = testError(sgemm_v2, gridDim, blockDim, M, N, K);
        printf("Max Error = %f\n", max_error);
    }

 
    printf("\nKernel sgemm_v2\n");
    const int M_list[15] = {128, 192, 256, 384, 512, 768, 1024, 1536, 2048, 3072, 4096, 6144, 8192, 12288, 16384};
    const int N_list[15] = {128, 192, 256, 384, 512, 768, 1024, 1536, 2048, 3072, 4096, 6144, 8192, 12288, 16384};
    const int K_list[15] = {1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024};


    const int TESTNUM = 15;
    for (int i = 0; i < TESTNUM; ++i) {
        const int M = M_list[i], N = N_list[i], K = K_list[i];

        dim3 blockDim(BN / TN, BM / TM);
        dim3 gridDim((N + BN - 1) / BN, (M + BM - 1) / BM);

        double max_sec = 0.0;
        double min_sec = DBL_MAX;
        double total_sec = 0.0;

        for (int j = 0; j < outer_repeat; ++j) {
            double this_sec = testPerformance(sgemm_v2, gridDim, blockDim, M, N, K, inner_repeat);
            max_sec = max(max_sec, this_sec);
            min_sec = min(min_sec, this_sec);
            total_sec += this_sec;
        }

        double avg_sec = total_sec / outer_repeat;
        double avg_Gflops = ((double)M) * N * K * 2/ 1024 / 1024 / 1024 / avg_sec;

        printf("M N K = %6d %6d %6d, Time = %12.8lf %12.8lf %12.8lf s, AVG Performance = %10.4lf Gflops\n",
                M, N, K, min_sec, avg_sec, max_sec, avg_Gflops);
    }
    return 0;
}

float testError(Sgemm gpuSgemm, dim3 gridDim, dim3 blockDim, const int M, const int N, const int K) {
    size_t size_a = M * K * sizeof(float);
    size_t size_b = K * N * sizeof(float);
    size_t size_c = M * N * sizeof(float);

    float *h_a, *h_b, *h_c, *d_a, *d_b, *d_c, *h_d_c;

    h_a = (float *)malloc(size_a);
    h_b = (float *)malloc(size_b);
    h_c = (float *)malloc(size_c);   // used for cpu
    h_d_c = (float *)malloc(size_c); // used for gpu --> cpu
    CHECK(cudaMalloc(&d_a, size_a));
    CHECK(cudaMalloc(&d_b, size_b));
    CHECK(cudaMalloc(&d_c, size_c));

    srand(time(0));
    for (int i = 0; i < M * K; ++i) {
        h_a[i] = rand() / float(RAND_MAX);
    }
    for (int i = 0; i < K * N; ++i) {
        h_b[i] = rand() / float(RAND_MAX);
    }
    CHECK(cudaMemset(d_c, 0xf, size_c));

    cpuSgemm(h_a, h_b, h_c, M, N, K);

    CHECK(cudaMemcpy(d_a, h_a, size_a, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_b, h_b, size_b, cudaMemcpyHostToDevice));
    gpuSgemm<<<gridDim, blockDim>>>(d_a, d_b, d_c, M, N, K);
    CHECK(cudaDeviceSynchronize());
    cudaError_t err = cudaPeekAtLastError();
    if (err != cudaSuccess) printf("testError: Kernel launch error: %s\n", cudaGetErrorString(err));
    CHECK(cudaMemcpy(h_d_c, d_c, size_c, cudaMemcpyDeviceToHost));

    float max_error = 0.0;
    for (int i = 0; i < M * N; i++) {
        float this_error = abs(h_d_c[i] - h_c[i]);
        if (this_error != this_error || max_error != max_error) {
            max_error = -NAN;
        } else {
            max_error = max(max_error, this_error);
        }
    }

    free(h_a);
    free(h_b);
    free(h_c);
    free(h_d_c);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return max_error;
}

float testPerformance(Sgemm gpuSgemm, dim3 gridDim, dim3 blockDim, const int M, const int N, const int K, const int repeat) {
    size_t size_a = M * K * sizeof(float);
    size_t size_b = K * N * sizeof(float);
    size_t size_c = M * N * sizeof(float);

    float *d_a, *d_b, *d_c;
    CHECK(cudaMalloc(&d_a, size_a));
    CHECK(cudaMalloc(&d_b, size_b));
    CHECK(cudaMalloc(&d_c, size_c));

    cudaEvent_t start, end;
    CHECK(cudaEventCreate(&start));
    CHECK(cudaEventCreate(&end));
    CHECK(cudaEventRecord(start));
    for (int i = 0; i < repeat; ++i) {
        gpuSgemm<<<gridDim, blockDim>>>(d_a, d_b, d_c, M, N, K);
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("Kernel launch failed: %s\n", cudaGetErrorString(err));
        }
    }
    CHECK(cudaEventRecord(end));
    CHECK(cudaEventSynchronize(end));

    float msec, sec;
    cudaEventElapsedTime(&msec, start, end);
    sec = msec / 1000.0 / repeat;

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    cudaEventDestroy(start);
    cudaEventDestroy(end);

    return sec;
}