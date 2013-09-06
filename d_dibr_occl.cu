#ifndef D_DIBR_OCCL_KERNEL 
#define D_DIBR_OCCL_KERNEL
#include "d_dibr_occl.h"
#include "d_mux_common.h"
#include "cuda_utils.h"
#include <math.h>

__global__ void dibr_occl_kernel(unsigned char *occl, float *disp,
                                 int dir,
                                 int num_rows, int num_cols)
{
    int tx = threadIdx.x + blockIdx.x * blockDim.x;
    int ty = threadIdx.y + blockIdx.y * blockDim.y;
    
    if ((tx > num_cols - 1) || (ty > num_rows - 1))
        return;
    
    int sd = (disp[tx + ty * num_cols] * dir); 
    int sx = min(max(tx + sd, 0), num_cols - 1);
    
    occl[sx + ty * num_cols] = 1;
}

void d_dibr_occl(unsigned char* d_occl_l, unsigned char* d_occl_r,
                 float* d_disp_l, float* d_disp_r,
                 int num_rows, int num_cols)
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
    
    checkCudaError(cudaMemset(d_occl_l, 0, sizeof(unsigned char) * num_rows * num_cols));
    checkCudaError(cudaMemset(d_occl_r, 0, sizeof(unsigned char) * num_rows * num_cols));

    /////////////////// 
    // KERNEL LAUNCH //
    ///////////////////
    
    dibr_occl_kernel<<<grid_sz, block_sz>>>(d_occl_r, d_disp_l, 1, num_rows, num_cols);
    dibr_occl_kernel<<<grid_sz, block_sz>>>(d_occl_l, d_disp_r, -1, num_rows, num_cols);
    cudaDeviceSynchronize(); 
}


void dibr_occl(unsigned char* occl_l, unsigned char* occl_r,
               float* disp_l, float* disp_r,
               int num_rows, int num_cols)
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
    
    /////////////////////// 
    // MEMORY ALLOCATION //
    ///////////////////////
    
    unsigned char* d_occl_l, *d_occl_r; 

    checkCudaError(cudaMalloc(&d_occl_l, sizeof(unsigned char) * num_rows * num_cols));
    checkCudaError(cudaMalloc(&d_occl_r, sizeof(unsigned char) * num_rows * num_cols));
    
    checkCudaError(cudaMemset(d_occl_l, 0, sizeof(unsigned char) * num_rows * num_cols));
    checkCudaError(cudaMemset(d_occl_r, 0, sizeof(unsigned char) * num_rows * num_cols));

    float* d_disp_l, *d_disp_r;

    checkCudaError(cudaMalloc(&d_disp_l, sizeof(float) * num_rows * num_cols));
    checkCudaError(cudaMalloc(&d_disp_r, sizeof(float) * num_rows * num_cols));

    checkCudaError(cudaMemcpy(d_disp_l, disp_l, sizeof(float) * num_rows * num_cols, cudaMemcpyHostToDevice));
    checkCudaError(cudaMemcpy(d_disp_r, disp_r, sizeof(float) * num_rows * num_cols, cudaMemcpyHostToDevice));
    
    /////////////////// 
    // KERNEL LAUNCH //
    ///////////////////
    
    startCudaTimer(&timer);
    dibr_occl_kernel<<<grid_sz, block_sz>>>(d_occl_r, d_disp_l, 1, num_rows, num_cols);
    stopCudaTimer(&timer, "Dis-occlusion Kernel");
    
    startCudaTimer(&timer);
    dibr_occl_kernel<<<grid_sz, block_sz>>>(d_occl_l, d_disp_r, -1, num_rows, num_cols);
    stopCudaTimer(&timer, "Dis-occlusion Kernel");

    checkCudaError(cudaMemcpy(occl_l, d_occl_l, sizeof(unsigned char) * num_rows * num_cols, cudaMemcpyDeviceToHost));
    checkCudaError(cudaMemcpy(occl_r, d_occl_r, sizeof(unsigned char) * num_rows * num_cols, cudaMemcpyDeviceToHost));

    cudaFree(d_occl_l);
    cudaFree(d_occl_r);
    cudaFree(d_disp_l);
    cudaFree(d_disp_r);
}

#endif
