#ifndef D_CI_AD_KERNEL 
#define D_CI_AD_KERNEL
#include "d_ci_ad.h"
#include "cuda_utils.h"
#include <math.h>

__global__ void ci_ad_kernel(unsigned char* img_l, unsigned char* img_r, 
                             float** cost_l, float** cost_R, int num_disp, int zero_disp,
                             int num_rows, int num_cols, int elem_sz,
                             int sm_w, int sm_sz)
{
    int gx = threadIdx.x + blockIdx.x * blockDim.x;
    int gy = threadIdx.y + blockIdx.y * blockDim.y;

    if ((gx > num_cols - 1) || (gy > num_rows - 1))
        return;
    
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    // Load Shared Memory
    extern __shared__ unsigned char sm_img[];
    unsigned char* sm_img_l = sm_img;
    unsigned char* sm_img_r = &sm_img[sm_sz];
    
    int gsx_lim = min(blockIdx.x * blockDim.x - zero_disp + 1 + sm_w, num_cols - 1);
    int gsx_init = max(gx - zero_disp + 1, 0);

    int ty_smw = ty * sm_w * elem_sz;
    int gy_numcols = gy * num_cols;

    for (int gsx = gsx_init, tsx = tx; gsx < gsx_lim; gsx += blockDim.x, tsx += blockDim.x)
    {
        int sidx = tsx * elem_sz + ty_smw;
        int gidx = (gsx + gy_numcols) * elem_sz;
        sm_img_l[sidx] = img_l[gidx];
        sm_img_l[sidx + 1] = img_l[gidx + 1];
        sm_img_l[sidx + 2] = img_l[gidx + 2];
        
        sm_img_r[sidx] = img_r[gidx];
        sm_img_r[sidx + 1] = img_r[gidx + 1];
        sm_img_r[sidx + 2] = img_r[gidx + 2];
    }

    __syncthreads();

    for (int d = 0; d < num_disp; ++d)
    {
        int r_coord = tx - 1 + d; 
        int ll = (tx + zero_disp - 1) * elem_sz + ty_smw;
        int lr = r_coord * elem_sz + ty_smw;
        float cost_b = abs(sm_img_l[ll] - sm_img_r[lr]);
        float cost_g = abs(sm_img_l[ll + 1] - sm_img_r[lr + 1]);
        float cost_r = abs(sm_img_l[ll + 2] - sm_img_r[lr + 2]);
        float cost = (cost_b + cost_g + cost_r) * 0.33333333333333333;
        cost_l[d][gx + gy_numcols] = cost;
        int gr_coord = min(max(gx + (d - zero_disp), 0), num_cols - 1);
        cost_R[d][gr_coord + gy_numcols] = cost;
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

    int sm_w = bw + num_disp - 1;
    int sm_sz = sm_w * bh * elem_sz;

    // Launch Kernel
    startCudaTimer(&timer);
    ci_ad_kernel<<<grid_sz, block_sz, 2 * sm_sz * sizeof(unsigned char)>>>(d_img_l, d_img_r, d_cost_l, d_cost_r, num_disp, zero_disp, num_rows, num_cols, elem_sz, sm_w, sm_sz);
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
