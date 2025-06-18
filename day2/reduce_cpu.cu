#include "common.h"

#ifdef USE_DP
    typedef double real;
#else
    typedef float real;
#endif

const int NUM_REPEATS = 20;
void timing(const real *x, const int N);
real reduce(const real *x, const int N);

int main()
{
    const int N = 100000000;
    real *x = (real *)malloc(sizeof(real) * N);
    for (int n = 0; n < N; ++n) {
        x[n] = 1.35;
    }

    timing(x, N);
    
    free(x);
    return 0;
}

void timing(const real *x, const int N)
{
    real sum = 0;

    for (int i = 0; i < NUM_REPEATS; ++i) {
        cudaEvent_t start, end;
        CHECK(cudaEventCreate(&start));
        CHECK(cudaEventCreate(&end));
        CHECK(cudaEventRecord(start));
        cudaEventQuery(start);

        sum = reduce(x, N);

        CHECK(cudaEventRecord(end));
        CHECK(cudaEventSynchronize(end));

        float elpased_time;
        CHECK(cudaEventElapsedTime(&elpased_time, start, end));
        printf("Time = %g ms.\n", elpased_time);

        CHECK(cudaEventDestroy(start));
        CHECK(cudaEventDestroy(end));
    }

    printf("sum = %f.\n", sum);
}

real reduce(const real *x, const int N)
{
    real sum = 0.0;
    for (int i = 0; i < N; ++i) {
        sum += x[i];
    }
    return sum;
}