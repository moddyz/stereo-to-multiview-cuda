#ifndef D_DR_INTERP_KERNEL 
#define D_DR_INTERP_KERNEL
#include "d_dr_irv.h"
#include "cuda_utils.h"
#include <math.h>



__global__ void dr_interp_kernel(float* disp, unsigned char* outliers,
                                 int num_rows, int num_cols, 
                                 int num_disp, int zero_disp)
{
    int gx = threadIdx.x + blockIdx.x * blockDim.x;
    int gy = threadIdx.y + blockIdx.y * blockDim.y;
    
    if ((gx > num_cols - 1) || (gy > num_rows - 1))
        return;



}
                                  

void dr_interp(float* disp, unsigned char* outliers, 
               int num_rows, int num_cols, int num_disp, int zero_disp)
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
    
    float* d_disp;

    checkCudaError(cudaMalloc(&d_disp, sizeof(float) * num_rows * num_cols));
    checkCudaError(cudaMemcpy(d_disp, disp, sizeof(float) * num_rows * num_cols, cudaMemcpyHostToDevice));

    unsigned char* d_outliers;

    checkCudaError(cudaMalloc(&d_outliers, sizeof(unsigned char) * num_rows * num_cols));
    checkCudaError(cudaMemcpy(d_outliers, outliers, sizeof(unsigned char) * num_rows * num_cols, cudaMemcpyHostToDevice));
   
    startCudaTimer(&timer);
    stopCudaTimer(&timer, "Disparity Refinement Proper Interpolation Kernel");

    checkCudaError(cudaMemcpy(disp, d_disp, sizeof(float) * num_rows * num_cols, cudaMemcpyDeviceToHost));

    cudaFree(d_disp);
    cudaFree(d_outliers);
    free(h_cross);
}

#endif
