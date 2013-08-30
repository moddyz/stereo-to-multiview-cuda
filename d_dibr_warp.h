#ifndef D_DIBR_WARP_H
#define D_DIBR_WARP_H
#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

__global__ void dibr_forward_map_kernel(unsigned char* img_out, unsigned char* img_in, int* disp_in,
                                        float shift, int num_disp, int zero_disp, 
                                        int num_rows, int num_cols, int elem_sz);


#endif
