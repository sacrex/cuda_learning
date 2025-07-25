__global__ void add2_kernel(float *c, const float* a, const float* b, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n) {
        c[i] = a[i] + b[i];
    }
    __syncthreads();
}

void launch_add2(float *c, const float* a, const float* b, int n) {
    dim3 grid( (n + 1023) / 1024);
    dim3 block(1024);

    add2_kernel<<<grid, block>>>(c, a, b, n);
}