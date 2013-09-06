#ifndef D_DIBR_BWARP_KERNEL 
#define D_DIBR_BWARP_KERNEL
#include "d_dibr_bwarp.h"#
#include "d_alu.h"
#include "d_mux_common.h"
#include "cuda_utils.h"
#include <math.h>

__global__ void dibr_backward_warp_kernel(unsigned char* img_out, unsigned char* img_in,
                                          unsigned char* occl, float *disp,
                                          float shift, int num_rows, int num_cols, int elem_sz)
{
    int tx = threadIdx.x + blockIdx.x * blockDim.x;
    int ty = threadIdx.y + blockIdx.y * blockDim.y;
    
    if ((tx > num_cols - 1) || (ty > num_rows - 1))
        return;

    if (occl[tx + ty * num_cols] == 0)
        return;
    
    float sd = (disp[tx + ty * num_cols] * shift); 
    int sx = min(max((float) tx + sd, 0.0f), (float)(num_cols - 1));

    img_out[(tx + ty * num_cols) * elem_sz] = alu_bilinear_interp(img_in, elem_sz, 0, sx, (float) ty, num_cols, num_rows);
    img_out[(tx + ty * num_cols) * elem_sz + 1] = alu_bilinear_interp(img_in, elem_sz, 1, sx, (float)ty, num_cols, num_rows);
    img_out[(tx + ty * num_cols) * elem_sz + 2] = alu_bilinear_interp(img_in, elem_sz, 2, sx, (float)ty, num_cols, num_rows);
}

void d_dibr_dbm(unsigned char* d_img_out,
                unsigned char* d_img_in_l, unsigned char* d_img_in_r, 
                float* d_disp_l, float* d_disp_r,
                unsigned char *d_occl_l, unsigned char *d_occl_r,
                float shift, int num_rows, int num_cols, int elem_sz)
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
    
    //////////// 
    // KERNEL //
    ////////////
    unsigned char* d_img_out_r; 
    checkCudaError(cudaMalloc(&d_img_out_r, sizeof(unsigned char) * num_rows * num_cols * elem_sz));
    
    checkCudaError(cudaMemset(d_img_out, 0, sizeof(unsigned char) * num_rows * num_cols * elem_sz));
    checkCudaError(cudaMemset(d_img_out_r, 0, sizeof(unsigned char) * num_rows * num_cols * elem_sz));

    dibr_backward_warp_kernel<<<grid_sz, block_sz>>>(d_img_out, d_img_in_l, d_occl_r, d_disp_r, -shift, num_rows, num_cols, elem_sz);   
    dibr_backward_warp_kernel<<<grid_sz, block_sz>>>(d_img_out_r, d_img_in_r, d_occl_l, d_disp_l, 1.0 - shift, num_rows, num_cols, elem_sz);
    cudaDeviceSynchronize();
    
    mux_merge_AB_kernel<<<grid_sz, block_sz>>>(d_img_out, d_img_out_r, d_occl_r, num_rows, num_cols, elem_sz);  
    cudaDeviceSynchronize(); 

    cudaFree(d_img_out_r);
}


void dibr_dbm(unsigned char* img_out,
              unsigned char* img_in_l, unsigned char* img_in_r, 
              float* disp_l, float* disp_r,
              unsigned char *occl_l, unsigned char *occl_r,
              float shift, int num_rows, int num_cols, int elem_sz)
{
    cudaEventPair_t timer;
    
    /////////////////////// 
    // DEVICE PARAMETERS //
    ///////////////////////
    
    size_t bw = 32;
    size_t bh = 32;
    size_t gw = (num_cols + bw - 1) / bw;
    size_t gh = (num_rows + bh - 1) / bh;
    const dim3 block_sz(bw, bh, 1);
    const dim3 grid_sz(gw, gh, 1);
    
    /////////////// 
    // OCCLUSION //
    ///////////////
    unsigned char* d_occl_l, *d_occl_r; 

    checkCudaError(cudaMalloc(&d_occl_l, sizeof(unsigned char) * num_rows * num_cols));
    checkCudaError(cudaMalloc(&d_occl_r, sizeof(unsigned char) * num_rows * num_cols));
    
    checkCudaError(cudaMemcpy(d_occl_l, occl_l, sizeof(unsigned char) * num_rows * num_cols, cudaMemcpyHostToDevice));
    checkCudaError(cudaMemcpy(d_occl_r, occl_r, sizeof(unsigned char) * num_rows * num_cols, cudaMemcpyHostToDevice));
    
    /////////////////////// 
    // MEMORY ALLOCATION //
    ///////////////////////
    float* d_disp_l, *d_disp_r;

    checkCudaError(cudaMalloc(&d_disp_l, sizeof(float) * num_rows * num_cols));
    checkCudaError(cudaMalloc(&d_disp_r, sizeof(float) * num_rows * num_cols));

    checkCudaError(cudaMemcpy(d_disp_l, disp_l, sizeof(float) * num_rows * num_cols, cudaMemcpyHostToDevice));
    checkCudaError(cudaMemcpy(d_disp_r, disp_r, sizeof(float) * num_rows * num_cols, cudaMemcpyHostToDevice));
    
    unsigned char* d_img_in_l, *d_img_in_r; 

    checkCudaError(cudaMalloc(&d_img_in_l, sizeof(unsigned char) * num_rows * num_cols * elem_sz));
    checkCudaError(cudaMalloc(&d_img_in_r, sizeof(unsigned char) * num_rows * num_cols * elem_sz));

    checkCudaError(cudaMemcpy(d_img_in_l, img_in_l, sizeof(unsigned char) * num_rows * num_cols * elem_sz, cudaMemcpyHostToDevice));
    checkCudaError(cudaMemcpy(d_img_in_r, img_in_r, sizeof(unsigned char) * num_rows * num_cols * elem_sz, cudaMemcpyHostToDevice));
    
    unsigned char* d_img_out_l, *d_img_out_r; 
    
    checkCudaError(cudaMalloc(&d_img_out_l, sizeof(unsigned char) * num_rows * num_cols * elem_sz));
    checkCudaError(cudaMalloc(&d_img_out_r, sizeof(unsigned char) * num_rows * num_cols * elem_sz));
    
    checkCudaError(cudaMemset(d_img_out_l, 0, sizeof(unsigned char) * num_rows * num_cols * elem_sz));
    checkCudaError(cudaMemset(d_img_out_r, 0, sizeof(unsigned char) * num_rows * num_cols * elem_sz));

    startCudaTimer(&timer);
    dibr_backward_warp_kernel<<<grid_sz, block_sz>>>(d_img_out_l, d_img_in_l, d_occl_r, d_disp_r, -shift, num_rows, num_cols, elem_sz);  
    stopCudaTimer(&timer, "DIBR Backward Map Kernel");
    
    startCudaTimer(&timer);
    dibr_backward_warp_kernel<<<grid_sz, block_sz>>>(d_img_out_r, d_img_in_r, d_occl_l, d_disp_l, 1.0 - shift, num_rows, num_cols, elem_sz);  
    stopCudaTimer(&timer, "DIBR Backward Map Kernel");
    
    startCudaTimer(&timer);
    mux_merge_AB_kernel<<<grid_sz, block_sz>>>(d_img_out_l, d_img_out_r, d_occl_r, num_rows, num_cols, elem_sz);  
    stopCudaTimer(&timer, "Merge Kernel");
    
    ///////////////// 
    // MEMORY COPY //
    /////////////////

    checkCudaError(cudaMemcpy(img_out, d_img_out_l, sizeof(unsigned char) * num_rows * num_cols * elem_sz, cudaMemcpyDeviceToHost));

    /////////////////// 
    // DE-ALLOCATION //
    ///////////////////

    cudaFree(d_disp_l);
    cudaFree(d_disp_r);
    cudaFree(d_img_in_l);
    cudaFree(d_img_in_r);
    cudaFree(d_img_out_l);
    cudaFree(d_img_out_r);
    cudaFree(d_occl_l);
    cudaFree(d_occl_r);
    cudaFree(d_img_out_r);
}

#endif

