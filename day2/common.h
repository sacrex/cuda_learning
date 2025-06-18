#pragma once
#include <stdio.h>

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
