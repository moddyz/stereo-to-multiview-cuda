#ifndef D_DC_WTA_KERNEL 
#define D_DC_WTA_KERNEL
#include "d_dc_wta.h"
#include "cuda_utils.h"
#include <math.h>
#include <float.h>
#include <limits.h>

__global__ void dc_wta_kernel(float** cost, float* disp, 
                              int num_disp, int zero_disp, 
                              int num_rows, int num_cols)
{
    int tx = threadIdx.x + blockIdx.x * blockDim.x;
    int ty = threadIdx.y + blockIdx.y * blockDim.y;
    
    if ((tx > num_cols - 1) || (ty > num_rows - 1))
        return;
    
    float lowest_cost = FLT_MAX;

    for (int d = 0; d < num_disp; ++d)
    {
       float current_cost = cost[d][tx + ty * num_cols];
       if (lowest_cost > current_cost)
       {
           lowest_cost = current_cost;
           disp[tx + ty * num_cols] = (float) (d - zero_disp);
       }
    }
}

void d_dc_wta(float** d_cost, float* d_disp, 
              int num_disp, int zero_disp, 
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
    
    //////////////////////
    // WINNER TAKES ALL //
    //////////////////////
    
    dc_wta_kernel<<<grid_sz, block_sz>>>(d_cost, d_disp, num_disp, zero_disp, num_rows, num_cols);
    cudaDeviceSynchronize(); 

}


void dc_wta(float** cost, float* disp, 
            int num_disp, int zero_disp, 
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

    float** d_cost;

    checkCudaError(cudaMalloc(&d_cost, sizeof(float*) * num_disp));

    float** h_cost = (float**) malloc(sizeof(float*) * num_disp);
    
    for (int d = 0; d < num_disp; ++d)
    {
        checkCudaError(cudaMalloc(&h_cost[d], sizeof(float) * num_rows * num_cols));
        checkCudaError(cudaMemcpy(h_cost[d], cost[d], sizeof(float) * num_rows * num_cols, cudaMemcpyHostToDevice));
    }

    checkCudaError(cudaMemcpy(d_cost, h_cost, sizeof(float*) * num_disp, cudaMemcpyHostToDevice));

    float* d_disp;
    checkCudaError(cudaMalloc(&d_disp, sizeof(float) * num_rows * num_cols));
    checkCudaError(cudaMemset(d_disp, 0, sizeof(float) * num_rows * num_cols));
    
    //////////////////////
    // WINNER TAKES ALL //
    //////////////////////
    
    startCudaTimer(&timer); 
    dc_wta_kernel<<<grid_sz, block_sz>>>(d_cost, d_disp, num_disp, zero_disp, num_rows, num_cols);
    stopCudaTimer(&timer, "Disparity Computation Kernel"); 
    
    checkCudaError(cudaMemcpy(disp, d_disp, sizeof(float) * num_rows * num_cols, cudaMemcpyDeviceToHost));
    
    ///////////////////
    // DE-ALLOCATION //
    ///////////////////
    
    cudaFree(d_cost);
    cudaFree(d_disp);
    for (int d = 0; d < num_disp; ++d)
        cudaFree(h_cost[d]);
    free(h_cost);
}

#endif
