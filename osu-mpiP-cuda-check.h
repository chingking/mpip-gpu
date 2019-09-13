#ifndef _OSU_MPIP_CUDA_
#define _OSU_MPIP_CUDA_

#ifdef ENABLE_CUDA_MPI

#include "cuda.h"
#include "cuda_runtime.h"

#define CUDA_CHECK(stmt)                                \
do {                                                    \
    cudaError_t result = (stmt);                        \
    if (cudaSuccess != result) {                        \
        PRINT_ERROR("[%s:%d] cuda failed with %s \n",   \
         __FILE__, __LINE__,cudaGetErrorString(result));\
        exit(EXIT_FAILURE);                             \
    }                                                   \
    MPIU_Assert(cudaSuccess == result);                 \
} while (0)

#define CU_CHECK(stmt)                                      \
do {                                                        \
    CUresult result = (stmt);                               \
    const char *err_str;                                    \
    if (CUDA_SUCCESS != result) {                           \
        cuGetErrorString(result, &err_str);                 \
        PRINT_ERROR("[%s:%d] cuda failed with %d (%s) \n",  \
         __FILE__, __LINE__,result, err_str);               \
        exit(EXIT_FAILURE);                                 \
    }                                                       \
    MPIU_Assert(CUDA_SUCCESS == result);                    \
} while (0)

int is_cuda_buffer(const void *buf) {
    int is_cuda_buf = 0;
    struct cudaPointerAttributes attr;
    cudaError_t cuda_err = cudaPointerGetAttributes(&attr, buf);
#if CUDA_VERSION >= 10000
    if (cuda_err == CUDA_SUCCESS && attr.type == cudaMemoryTypeDevice)
#else
    if (cuda_err == CUDA_SUCCESS && attr.memoryType == cudaMemoryTypeDevice)
#endif
    {
        is_cuda_buf = 1;
        //fprintf(stderr, "detected device buffer\n");
    }
    return is_cuda_buf;
}

#endif /* ENABLE_CUDA_MPI */
#endif /* _OSU_MPIP_CUDA_ */
