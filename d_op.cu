#ifndef D_OP_KERNEL 
#define D_OP_KERNEL
#include "d_op.h"
#include "cuda_utils.h"
#include <math.h>

__global__ void op_invertnormf_kernel(float *values, int num_rows, int num_cols)
{
    int tx = threadIdx.x + blockIdx.x * blockDim.x;
    int ty = threadIdx.y + blockIdx.y * blockDim.y;
    
    if ((tx > num_cols - 1) || (ty > num_rows - 1))
        return;
    
    values[tx + ty * num_cols] = 1.0f - values[tx + ty * num_cols];
}

#endif
