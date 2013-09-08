#ifndef D_OP_H
#define D_OP_H
#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

__global__ void op_invertnormf_kernel(float *values, int num_rows, int num_cols);
                                   

#endif
