#ifndef D_CI_AD_KERNEL 
#define D_CI_AD_KERNEL
#include "d_ci_ad.h"
#include "cuda_utils.h"
#include <math.h>

__global__ void ci_ad_kernel(unsigned char* img_l, unsigned char* img_r, float** cost_l, 
                            float** cost_R, int num_disp, int zero_disp, int num_rows, int num_cols, int elem_sz)
{
    int tx = threadIdx.x + blockIdx.x * blockDim.x;
    int ty = threadIdx.y + blockIdx.y * blockDim.y;
    
    if ((tx > num_cols - 1) || (ty > num_rows - 1))
        return;

    for (int d = 0; d < num_disp; ++d)
    {
        int r_coord = min(max(tx + (d - zero_disp), 0), num_cols - 1);
        int ll = (tx + ty * num_cols) * elem_sz;
        int lr = (r_coord + ty * num_cols) * elem_sz;
        float cost_b = (float) abs(img_l[ll] - img_r[lr]);
        float cost_g = (float) abs(img_l[ll + 1] - img_r[lr + 1]);
        float cost_r = (float) abs(img_l[ll + 2] - img_r[lr + 2]);
        float cost = (cost_b + cost_g + cost_r) / 3.0;
        cost_l[d][tx + ty * num_cols] = cost;
        cost_R[d][r_coord + ty * num_cols] = cost;
    }
}

void ci_ad(unsigned char* img_l, unsigned char* img_r, float** cost_l, float** cost_r, int num_disp, int zero_disp, int num_rows, int num_cols, int elem_sz)
{
    cudaEventPair_t timer;
    
    // Device Memory Allocation & Copy
    unsigned char* d_img_l;
    unsigned char* d_img_r;

    checkCudaError(cudaMalloc(&d_img_l, sizeof(unsigned char) * num_rows * num_cols * elem_sz));
    checkCudaError(cudaMemcpy(d_img_l, img_l, sizeof(unsigned char) * num_rows * num_cols * elem_sz, cudaMemcpyHostToDevice));

    checkCudaError(cudaMalloc(&d_img_r, sizeof(unsigned char) * num_rows * num_cols * elem_sz));
    checkCudaError(cudaMemcpy(d_img_r, img_r, sizeof(unsigned char) * num_rows * num_cols * elem_sz, cudaMemcpyHostToDevice));

    // Device Cost Memory
    float** d_cost_l;
    float** d_cost_r;
    
    checkCudaError(cudaMalloc(&d_cost_l, sizeof(float*) * num_disp));
    checkCudaError(cudaMalloc(&d_cost_r, sizeof(float*) * num_disp));
    
    float** h_cost_l = (float**) malloc(sizeof(float*) * num_disp);
    float** h_cost_r = (float**) malloc(sizeof(float*) * num_disp);
    
    for (int d = 0; d < num_disp; ++d)
    {
        checkCudaError(cudaMalloc(&h_cost_l[d], sizeof(float) * num_rows * num_cols));
        checkCudaError(cudaMalloc(&h_cost_r[d], sizeof(float) * num_rows * num_cols));
    }
    
    checkCudaError(cudaMemcpy(d_cost_l, h_cost_l, sizeof(float*) * num_disp, cudaMemcpyHostToDevice));
    checkCudaError(cudaMemcpy(d_cost_r, h_cost_r, sizeof(float*) * num_disp, cudaMemcpyHostToDevice));

	// Setup Block & Grid Size
    size_t bw = 32;
    size_t bh = 32;
    
    size_t gw = (num_cols + bw - 1) / bw;
    size_t gh = (num_rows + bh - 1) / bh;
    
    const dim3 block_sz(bw, bh, 1);
    const dim3 grid_sz(gw, gh, 1);

    // Launch Kernel
    startCudaTimer(&timer);
    ci_ad_kernel<<<grid_sz, block_sz>>>(d_img_l, d_img_r, d_cost_l, d_cost_r, num_disp, zero_disp, num_rows, num_cols, elem_sz);
    stopCudaTimer(&timer, "Cost Initialization - Absolute Difference Kernel");
    
    // Copy Device Data to Host
    for (int d = 0; d < num_disp; ++d)
    {
        checkCudaError(cudaMemcpy(cost_l[d], h_cost_l[d], sizeof(float) * num_rows * num_cols, cudaMemcpyDeviceToHost));
        checkCudaError(cudaMemcpy(cost_r[d], h_cost_r[d], sizeof(float) * num_rows * num_cols, cudaMemcpyDeviceToHost));
    }

    
    // Deallocation
    cudaFree(d_img_l);
    cudaFree(d_img_r);
    cudaFree(d_cost_l);
    cudaFree(d_cost_r);
    for (int d = 0; d < num_disp; ++d)
    {
        cudaFree(h_cost_l[d]);
        cudaFree(h_cost_r[d]);
    }
}

#endif
