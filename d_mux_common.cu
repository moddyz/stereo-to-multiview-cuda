#ifndef D_MUX_COMMON_KERNEL 
#define D_MUX_COMMON_KERNEL
#include "d_mux_common.h"
#include "cuda_utils.h"
#include <math.h>

__global__ void mux_merge_AB_kernel(unsigned char* img_b, unsigned char* img_a, unsigned char* mask_a,
                                    int num_rows, int num_cols, int elem_sz)
{
    int tx = threadIdx.x + blockIdx.x * blockDim.x;
    int ty = threadIdx.y + blockIdx.y * blockDim.y;
    
    if ((tx > num_cols - 1) || (ty > num_rows - 1))
        return;
    
    unsigned char maskval_a = mask_a[tx + ty * num_cols];
    if (maskval_a == 0)
    {
        unsigned char clr_a_b = img_a[(tx + ty * num_cols) * elem_sz];
        unsigned char clr_a_g = img_a[(tx + ty * num_cols) * elem_sz + 1];
        unsigned char clr_a_r = img_a[(tx + ty * num_cols) * elem_sz + 2];
        img_b[(tx + ty * num_cols) * elem_sz] = clr_a_b;
        img_b[(tx + ty * num_cols) * elem_sz + 1] = clr_a_g;
        img_b[(tx + ty * num_cols) * elem_sz + 2] = clr_a_r;
    }
}

#endif
