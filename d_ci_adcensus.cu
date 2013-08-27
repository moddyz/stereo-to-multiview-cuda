#ifndef D_CI_ADCENSUS_KERNEL 
#define D_CI_ADCENSUS_KERNEL
#include "d_ci_adcensus.h"
#include "d_ci_ad.h"
#include "d_ci_census.h"
#include "cuda_utils.h"
#include <math.h>

void ci_adcensus(unsigned char* img_l, unsigned char* img_r, float** cost_l, float** cost_r, int num_disp, 
               int zero_disp, int num_rows, int num_cols, int elem_sz)
{
    cudaEventPair_t timer;
    
    //////////// 
    // COMMON //
    ////////////
    
    // Device Image Memory Allocation
    unsigned char* d_img_l;
    unsigned char* d_img_r;

    checkCudaError(cudaMalloc(&d_img_l, sizeof(unsigned char) * num_rows * num_cols * elem_sz));
    checkCudaError(cudaMemcpy(d_img_l, img_l, sizeof(unsigned char) * num_rows * num_cols * elem_sz, cudaMemcpyHostToDevice));

    checkCudaError(cudaMalloc(&d_img_r, sizeof(unsigned char) * num_rows * num_cols * elem_sz));
    checkCudaError(cudaMemcpy(d_img_r, img_r, sizeof(unsigned char) * num_rows * num_cols * elem_sz, cudaMemcpyHostToDevice));
    
    // Setup Block & Grid Size
    size_t bw = 32;
    size_t bh = 32;
    size_t gw = (num_cols + bw - 1) / bw;
    size_t gh = (num_rows + bh - 1) / bh;
    const dim3 block_sz(bw, bh, 1);
    const dim3 grid_sz(gw, gh, 1);

    ////////
    // AD //
    ////////

    // Device Memory Allocation & Copy
    float** d_ad_cost_l;
    float** d_ad_cost_r;
    
    checkCudaError(cudaMalloc(&d_ad_cost_l, sizeof(float*) * num_disp));
    checkCudaError(cudaMalloc(&d_ad_cost_r, sizeof(float*) * num_disp));
    
    float** h_ad_cost_l = (float**) malloc(sizeof(float*) * num_disp);
    float** h_ad_cost_r = (float**) malloc(sizeof(float*) * num_disp);
    
    for (int d = 0; d < num_disp; ++d)
    {
        checkCudaError(cudaMalloc(&h_ad_cost_l[d], sizeof(float) * num_rows * num_cols));
        checkCudaError(cudaMalloc(&h_ad_cost_r[d], sizeof(float) * num_rows * num_cols));
    }
    
    checkCudaError(cudaMemcpy(d_ad_cost_l, h_ad_cost_l, sizeof(float*) * num_disp, cudaMemcpyHostToDevice));
    checkCudaError(cudaMemcpy(d_ad_cost_r, h_ad_cost_r, sizeof(float*) * num_disp, cudaMemcpyHostToDevice));
    
    // Launch Kernel
    startCudaTimer(&timer);
    ci_ad_kernel<<<grid_sz, block_sz>>>(d_img_l, d_img_r, d_ad_cost_l, d_ad_cost_r, num_disp, zero_disp, num_rows, num_cols, elem_sz);
    stopCudaTimer(&timer, "Cost Initialization - Absolute Difference Kernel");

    ////////////
    // CENSUS //
    ////////////

    // Device Memory Allocation & Copy
    unsigned long long* d_census_l;
    unsigned long long* d_census_r;

    checkCudaError(cudaMalloc(&d_census_l, sizeof(unsigned long long) * num_rows * num_cols * elem_sz));
    checkCudaError(cudaMalloc(&d_census_r, sizeof(unsigned long long) * num_rows * num_cols * elem_sz));

    
    // Launch Census Transform Kernel
    startCudaTimer(&timer);
    tx_census_9x7_kernel<<<grid_sz, block_sz>>>(d_img_l, d_census_l, num_rows, num_cols, elem_sz);
    stopCudaTimer(&timer, "Census Transform Kernel");
    
    startCudaTimer(&timer);
    tx_census_9x7_kernel<<<grid_sz, block_sz>>>(d_img_r, d_census_r, num_rows, num_cols, elem_sz);
    stopCudaTimer(&timer, "Census Transform Kernel");
    
    // Cost Initialization Device Cost Memory
    float** d_census_cost_l;
    float** d_census_cost_r;
    
    checkCudaError(cudaMalloc(&d_census_cost_l, sizeof(float*) * num_disp));
    checkCudaError(cudaMalloc(&d_census_cost_r, sizeof(float*) * num_disp));
    
    float** h_census_cost_l = (float**) malloc(sizeof(float*) * num_disp);
    float** h_census_cost_r = (float**) malloc(sizeof(float*) * num_disp);
    
    for (int d = 0; d < num_disp; ++d)
    {
        checkCudaError(cudaMalloc(&h_census_cost_l[d], sizeof(float) * num_rows * num_cols));
        checkCudaError(cudaMalloc(&h_census_cost_r[d], sizeof(float) * num_rows * num_cols));
    }
    
    checkCudaError(cudaMemcpy(d_census_cost_l, h_census_cost_l, sizeof(float*) * num_disp, cudaMemcpyHostToDevice));
    checkCudaError(cudaMemcpy(d_census_cost_r, h_census_cost_r, sizeof(float*) * num_disp, cudaMemcpyHostToDevice));
    
    // Launch Cost Kernel
    startCudaTimer(&timer);
    ci_census_kernel<<<grid_sz, block_sz>>>(d_census_r, d_census_r, d_census_cost_l, d_census_cost_r, num_disp, zero_disp, num_rows, num_cols, elem_sz);
    stopCudaTimer(&timer, "Census Cost Kernel");
    
    /////////////////
    // AD + CENSUS //
    /////////////////
    
    for (int d = 0; d < num_disp; ++d)
    {
        checkCudaError(cudaMemcpy(cost_l[d], h_census_cost_l[d], sizeof(float) * num_rows * num_cols, cudaMemcpyDeviceToHost));
        checkCudaError(cudaMemcpy(cost_r[d], h_census_cost_r[d], sizeof(float) * num_rows * num_cols, cudaMemcpyDeviceToHost));
    }

    // Device De-allocation
    cudaFree(d_img_l);
    cudaFree(d_img_r);
    cudaFree(d_census_l);
    cudaFree(d_census_r);
    cudaFree(d_census_cost_l);
    cudaFree(d_census_cost_r);
    cudaFree(d_ad_cost_l);
    cudaFree(d_ad_cost_r);
    for (int d = 0; d < num_disp; ++d)
    {
        cudaFree(h_census_cost_l[d]);
        cudaFree(h_census_cost_r[d]);
        cudaFree(h_ad_cost_l[d]);
        cudaFree(h_ad_cost_r[d]);
    }

    // Host De-allocation
    free(h_census_cost_l); 
    free(h_census_cost_r); 
    free(h_ad_cost_l); 
    free(h_ad_cost_r); 
}

#endif
