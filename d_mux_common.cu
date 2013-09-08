#ifndef D_MUX_COMMON_KERNEL 
#define D_MUX_COMMON_KERNEL
#include "d_mux_common.h"
#include "cuda_utils.h"
#include <math.h>

__global__ void mux_merge_AB_kernel(unsigned char* img_b, unsigned char* img_a, float* mask_a,
                                    int num_rows, int num_cols, int elem_sz)
{
    int tx = threadIdx.x + blockIdx.x * blockDim.x;
    int ty = threadIdx.y + blockIdx.y * blockDim.y;
    
    if ((tx > num_cols - 1) || (ty > num_rows - 1))
        return;
    
    float val_mask = mask_a[tx + ty * num_cols];
    //printf("%f ", val_mask);
    
    float clr_a_b = val_mask * (float) img_a[(tx + ty * num_cols) * elem_sz];
    float clr_a_g = val_mask * (float) img_a[(tx + ty * num_cols) * elem_sz + 1];
    float clr_a_r = val_mask * (float) img_a[(tx + ty * num_cols) * elem_sz + 2];
    
    float clr_b_b = (1.0f - val_mask) * (float) img_b[(tx + ty * num_cols) * elem_sz];
    float clr_b_g = (1.0f - val_mask) * (float) img_b[(tx + ty * num_cols) * elem_sz + 1];
    float clr_b_r = (1.0f - val_mask) * (float) img_b[(tx + ty * num_cols) * elem_sz + 2];
    
    img_b[(tx + ty * num_cols) * elem_sz] = (unsigned char) clr_b_b + (unsigned char) clr_a_b;
    img_b[(tx + ty * num_cols) * elem_sz + 1] = (unsigned char) clr_b_g + (unsigned char) clr_a_g;
    img_b[(tx + ty * num_cols) * elem_sz + 2] = (unsigned char) clr_b_r + (unsigned char) clr_a_r;
}

#endif
