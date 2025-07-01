# 文件说明 #

# sgemm_naive_block_tile.cu
使用block tiling技术，分块计算，使用global memory

# sgemm_v1.cu
使用 block tiling + thread tiling + share memory

# sgemm_v2.cu
使用 block tiling + thread tiling + share memory
并使用以下方法解决bank conflict:
1) 对A矩阵[BM,BK]从global memory加载到share memory, 进行转置操作,
此时每个线程对[BM,BK]的读取时，一个float4指令可以读取4个元素，这4个元素
在同一行；并且每个线程从A和B的share memory各读取8个元素，那么假如连续
读取8个元素，会出现bank conflict,为了解决这个问题，按如下方式：
把C矩阵的计算分成上下左右4个部分，这种情况下，C矩阵中的一个这样的矩阵[TM,TN]是
由一个线程计算的，而在这个矩阵的同一行中，共有16个线程(BM/TM),他们都同时读取
TM长的同一份数据，而TM=8,所以16个线程一组读取同样的8个元素。
也就是说：由于矩阵C[BM,BN]是一个线程块来来计算，而线程块中包含16*16个线程。
每个线程从BM中读取8个元素，从BN中读取8个元素。那么刚好可以根据(ty,tx)的线程索引来计算
这个线程从BM中读取的哪8个元素，并且上面的说法，这8个元素需要上面读一半+下面读一半（解决bank conflict）。
因此，可以通过ty(这个和BM绑定)来确定，例如ty=0，表示读取的是0~3和64~67这8个元素。上下各4个；
而ty=15就是60~63和124~127这8个元素。下面这个图说明了情况：
[thread-tiling](./thread_tiling.jpg)

其实就是按照线程的ty来计算就行，需要注意的是，这里由于A[BM,BK]在共享内存中是转置成[BK,BM],但是
我们在加载A[BM,BK]时,用的是blockIdx.y来对应BM轴,因此在共享内存中的[BK,BM]，此时用ty对应BM轴,
用tx对应BK轴。
因此下面的这段代码就可以看懂了。
```
FLOAT4(r_comp_a[0]) = FLOAT4(s_a[tk][ty * TM / 2         ]);
FLOAT4(r_comp_a[4]) = FLOAT4(s_a[tk][ty * TM / 2 + BM / 2]);
FLOAT4(r_comp_b[0]) = FLOAT4(s_b[tk][tx * TN / 2         ]);
FLOAT4(r_comp_b[4]) = FLOAT4(s_b[tk][tx * TN / 2 + BN / 2]);
```

最后说一点，由于计算的C[TM,TN]是一个8x8矩阵，这个矩阵其实算是4个4x4矩阵，分别来自上下左右4个矩阵中的对应位置。
因为最后还有这段代码，表示把结果写回到C中。
```
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
```
这段代码就是把r_c(这是一个寄存器存储的8x8矩阵),他写回到原C矩阵，就是按上下块写，并且在上块中，也是按左右块写(+BN / 2).
需要注意的是，这里需要对每次存储都进行边界判断，对于c_gemm_m和c_gemm_n分别判断, 对于c_gemm_addr + BN / 2的地址，因为这个位置
和c_gemm_addr处于同一行，所以他的c_gemm_n 要 小于 N - BN / 2,这样才不会越界。

一张图总结：
[compute](./thread_tiling_compute.jpg)

# sgemm_v3.cu
通过使用双buffer机制，实现流水并行 -- 读取+计算
1、load global data -> share memory
2、loop:
     switch share memory location
     load another global data -> register
     calc use first share memory data
     register -> second share memory
3、calc last share memory data