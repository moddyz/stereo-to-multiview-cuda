#ifndef D_DR_DCC_KERNEL
#define D_DR_DCC_KERNEL
#include "d_dr_dcc.h"
#include "cuda_utils.h"
#include <math.h>

/* 
   Disparity Dis-occlusion Check Kernel

 */

/* 
   Disparity Cross Check Kernel
   LD(p) == RD(p + LD(p))
   RD(p) == LD(p - RD(p))
 */

__global__ void dr_merge_errors_kernel(unsigned char* outliers_l, unsigned char* outliers_r,
                                       unsigned char* disoccl_l, unsigned char* disoccl_r,
                                       int num_rows, int num_cols)
{
    int gx = threadIdx.x + blockIdx.x * blockDim.x; 
    int gy = threadIdx.y + blockIdx.y * blockDim.y; 

    if (gx >= num_cols || gy >= num_rows)
        return;

    if (outliers_l[gx + gy * num_cols] == 1 &&  disoccl_l[gx + gy * num_cols] == 1)
        outliers_l[gx + gy * num_cols] = 2;
    
    if (outliers_r[gx + gy * num_cols] == 1 &&  disoccl_r[gx + gy * num_cols] == 1)
        outliers_r[gx + gy * num_cols] = 2;
}

__global__ void dr_ddc_kernel(unsigned char* disoccl_l, unsigned char *disoccl_r,
                             float *disp_l, float *disp_r,
                             int num_rows, int num_cols)
{
    int gx = threadIdx.x + blockIdx.x * blockDim.x; 
    int gy = threadIdx.y + blockIdx.y * blockDim.y; 

    if (gx >= num_cols || gy >= num_rows)
        return;
    
    int d = (int) disp_l[gx + gy * num_cols];
    int coord = min(max(gx + d, 0), num_cols - 1);
    
    disoccl_r[coord + gy * num_cols] = 0;
    
    d = (int) disp_r[gx + gy * num_cols];
    coord = min(max(gx - d, 0), num_cols - 1);
    
    disoccl_l[coord + gy * num_cols] = 0;
}
    

__global__ void dr_dcc_kernel(unsigned char* outliers_l, unsigned char *outliers_r,
                             float *disp_l, float *disp_r,
                             float thresh,
                             int num_rows, int num_cols)
{
    int gx = threadIdx.x + blockIdx.x * blockDim.x; 
    int gy = threadIdx.y + blockIdx.y * blockDim.y; 

    if (gx >= num_cols || gy >= num_rows)
        return;

    float d =  disp_l[gx + gy * num_cols];
    int coord = min(max(gx + (int) d, 0), num_cols - 1);
    float d_ref = disp_r[coord + gy * num_cols];
    
    if (abs(d - d_ref) > thresh)
        outliers_l[gx + gy * num_cols] = 1;
    
    d = disp_r[gx + gy * num_cols];
    coord = min(max(gx - (int) d, 0), num_cols - 1);
    d_ref = disp_l[coord + gy * num_cols];
    
    if (abs(d - d_ref) > thresh)
        outliers_r[gx + gy * num_cols] = 1;

}

void dr_dcc(unsigned char *outliers_l, unsigned char *outliers_r,
            float* disp_l, float *disp_r,
            int num_rows, int num_cols)
{
    cudaEventPair_t timer;
    
    /////////////////////// 
    // DEVICE PARAMETERS //
    ///////////////////////

    size_t bw = num_cols;
    size_t bh = 1;
    size_t gw = (num_cols + bw - 1) / bw;
    size_t gh = (num_rows + bh - 1) / bh;
    const dim3 block_sz(bw, bh, 1);
    const dim3 grid_sz(gw, gh, 1);
    
    ///////////////////////
    // MEMORY ALLOCATION //
    ///////////////////////

    float* d_disp_l;
    checkCudaError(cudaMalloc(&d_disp_l, sizeof(float) * num_rows * num_cols));
    checkCudaError(cudaMemcpy(d_disp_l, disp_l, sizeof(float) * num_rows * num_cols, cudaMemcpyHostToDevice));
    
    float* d_disp_r;
    checkCudaError(cudaMalloc(&d_disp_r, sizeof(float) * num_rows * num_cols));
    checkCudaError(cudaMemcpy(d_disp_r, disp_r, sizeof(float) * num_rows * num_cols, cudaMemcpyHostToDevice));

    unsigned char* d_disoccl_l;
    checkCudaError(cudaMalloc(&d_disoccl_l, sizeof(unsigned char) * num_rows * num_cols));
    checkCudaError(cudaMemset(d_disoccl_l, 1, sizeof(unsigned char) * num_rows * num_cols));
    
    unsigned char* d_disoccl_r;
    checkCudaError(cudaMalloc(&d_disoccl_r, sizeof(unsigned char) * num_rows * num_cols));
    checkCudaError(cudaMemset(d_disoccl_r, 1, sizeof(unsigned char) * num_rows * num_cols));

    unsigned char* d_outliers_l;
    checkCudaError(cudaMalloc(&d_outliers_l, sizeof(unsigned char) * num_rows * num_cols));
    checkCudaError(cudaMemset(d_outliers_l, 0, sizeof(unsigned char) * num_rows * num_cols));
    
    unsigned char* d_outliers_r;
    checkCudaError(cudaMalloc(&d_outliers_r, sizeof(unsigned char) * num_rows * num_cols));
    checkCudaError(cudaMemset(d_outliers_r, 0, sizeof(unsigned char) * num_rows * num_cols));
    
    //////////////////////
    // WINNER TAKES ALL //
    //////////////////////
    
    startCudaTimer(&timer); 
    dr_dcc_kernel<<<grid_sz, block_sz>>>(d_outliers_l, d_outliers_r, d_disp_l, d_disp_r, 1.0, num_rows, num_cols);
    stopCudaTimer(&timer, "Outliers Detection Kernel"); 

    startCudaTimer(&timer);
    dr_ddc_kernel<<<grid_sz, block_sz>>>(d_disoccl_l, d_disoccl_r, d_disp_l, d_disp_r, num_rows, num_cols);
    stopCudaTimer(&timer, "Disoccolusion Detection Kernel"); 

    startCudaTimer(&timer);
    dr_merge_errors_kernel<<<grid_sz, block_sz>>>(d_outliers_l, d_outliers_r, d_disoccl_l, d_disoccl_r, num_rows, num_cols);
    stopCudaTimer(&timer, "Error Merge Kernel");
    
    checkCudaError(cudaMemcpy(outliers_l, d_outliers_l, sizeof(unsigned char) * num_rows * num_cols, cudaMemcpyDeviceToHost));
    checkCudaError(cudaMemcpy(outliers_r, d_outliers_r, sizeof(unsigned char) * num_rows * num_cols, cudaMemcpyDeviceToHost));
    
    ///////////////////
    // DE-ALLOCATION //
    ///////////////////
    
    cudaFree(d_disp_l);
    cudaFree(d_disp_r);
    cudaFree(d_outliers_l);
    cudaFree(d_outliers_r);
}

#endif
