#ifndef D_DIBR_WARP_KERNEL 
#define D_DIBR_WARP_KERNEL
#include "d_dibr_warp.h"
#include "cuda_utils.h"
#include <math.h>

__global__ void dibr_forward_warp_kernel(unsigned char* img_out, unsigned char* holes, 
                                         unsigned char* img_in, int* disp,
                                         float shift, int num_disp, int zero_disp, 
                                         int num_rows, int num_cols, int elem_sz)
{
    int tx = threadIdx.x + blockIdx.x * blockDim.x;
    int ty = threadIdx.y + blockIdx.y * blockDim.y;
    
    if ((tx > num_cols - 1) || (ty > num_rows - 1))
        return;
    
    int sd = (int) ((float) disp[tx + ty * num_cols] * shift); 
    int sx = min(max(tx + sd, 0), num_cols - 1);

    img_out[(sx + ty * num_cols) * elem_sz] = img_in[(tx + ty * num_cols) * elem_sz];
    img_out[(sx + ty * num_cols) * elem_sz + 1] = img_in[(tx + ty * num_cols) * elem_sz + 1];
    img_out[(sx + ty * num_cols) * elem_sz + 2] = img_in[(tx + ty * num_cols) * elem_sz + 2];
    
    holes[sx + ty * num_cols] = 1;
}


void dibr_dfm(unsigned char* img_out_l, unsigned char* img_out_r, 
              unsigned char* img_in_l, unsigned char* img_in_r, int* disp_l, int* disp_r,
              float shift, int num_disp, int zero_disp, 
              int num_rows, int num_cols, int elem_sz)
{
    /////////////////////// 
    // DEVICE PARAMETERS //
    ///////////////////////
    
    size_t bw = 32;
    size_t bh = 32;
    size_t gw = (num_cols + bw - 1) / bw;
    size_t gh = (num_rows + bh - 1) / bh;
    const dim3 block_sz(bw, bh, 1);
    const dim3 grid_sz(gw, gh, 1);
    
    /////////////////////// 
    // MEMORY ALLOCATION //
    ///////////////////////
    int* d_disp_l, d_disp_r;

    checkCudaError(cudaMalloc(&d_disp_l, sizeof(int) * num_rows * num_cols));
    checkCudaError(cudaMalloc(&d_disp_r, sizeof(int) * num_rows * num_cols));
    
    unsigned char* d_img_in_l, d_img_in_r; 

    checkCudaError(cudaMalloc(&d_img_in_l, sizeof(unsigned char) * num_rows * num_cols * elem_sz));
    checkCudaError(cudaMalloc(&d_img_in_r, sizeof(unsigned char) * num_rows * num_cols * elem_sz));

    unsigned char* d_img_out_l, d_img_out_r; 
    
    checkCudaError(cudaMalloc(&d_img_out_l, sizeof(unsigned char) * num_rows * num_cols * elem_sz));
    checkCudaError(cudaMalloc(&d_img_out_r, sizeof(unsigned char) * num_rows * num_cols * elem_sz));
    
    

}


#endif
