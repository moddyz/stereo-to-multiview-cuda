#ifndef D_DIBR_WARP_KERNEL 
#define D_DIBR_WARP_KERNEL
#include "d_dibr_warp.h"
#include "cuda_utils.h"
#include <math.h>

__global__ void dibr_forward_warp_kernel(unsigned char* img_out, unsigned char* img_in, int* disp_in,
                                         float shift, int num_disp, int zero_disp, 
                                         int num_rows, int num_cols, int elem_sz);


#endif
