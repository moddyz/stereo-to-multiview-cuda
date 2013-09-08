#ifndef D_CI_ADCENSUS_KERNEL 
#define D_CI_ADCENSUS_KERNEL
#include "d_ci_adcensus.h"
#include "d_ci_ad.h"
#include "d_ci_census.h"
#include "cuda_utils.h"
#include <math.h>

__global__ void ci_adcensus_kernel(float** ad_cost_l, float** ad_cost_r, 
                                   float** census_cost_l, float** census_cost_r,
                                   float** adcensus_cost_l, float** adcensus_cost_r,
                                   float inv_ad_coeff, float inv_census_coeff, 
                                   int num_disp, int zero_disp, 
                                   int num_rows, int num_cols, int elem_sz)
{
    int tx = threadIdx.x + blockIdx.x * blockDim.x;
    int ty = threadIdx.y + blockIdx.y * blockDim.y;
    
    if ((tx > num_cols - 1) || (ty > num_rows - 1))
        return;

    int pdx = tx + ty * num_cols;

    for (int d = 0; d < num_disp; ++d)
    {
       float ad_comp_l = 1.0 - exp(-ad_cost_l[d][pdx]*inv_ad_coeff);
       float ad_comp_r = 1.0 - exp(-ad_cost_r[d][pdx]*inv_ad_coeff);

       float census_comp_l = 1.0 - exp(-census_cost_l[d][pdx]*inv_census_coeff);
       float census_comp_r = 1.0 - exp(-census_cost_r[d][pdx]*inv_census_coeff);
       
       adcensus_cost_l[d][pdx] = ad_comp_l + census_comp_l;
       adcensus_cost_r[d][pdx] = ad_comp_r + census_comp_r;
    }
}

void d_ci_adcensus(unsigned char* d_img_l, unsigned char* d_img_r, 
                 float** d_adcensus_cost_l, float** d_adcensus_cost_r, 
                 float** h_adcensus_cost_l, float** h_adcensus_cost_r, 
                 float *d_adcensus_cost_memory,
                 float ad_coeff, float census_coeff, int num_disp, int zero_disp, 
                 int num_rows, int num_cols, int elem_sz)
{
    // Setup Block & Grid Size
    size_t bw = 32;
    size_t bh = 32;
    size_t gw = (num_cols + bw - 1) / bw;
    size_t gh = (num_rows + bh - 1) / bh;
    const dim3 block_sz(bw, bh, 1);
    const dim3 grid_sz(gw, gh, 1);
    
    size_t img_sz = num_rows * num_cols;
    size_t imgelem_sz = img_sz * elem_sz;
    size_t cost_sz = img_sz * num_disp;
    
    
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

    float* d_ad_cost_memory;
    checkCudaError(cudaMalloc(&d_ad_cost_memory, sizeof(float) * cost_sz * 2));
    
    for (int d = 0; d < num_disp; ++d)
    {
        h_ad_cost_l[d] = d_ad_cost_memory + (d * img_sz);
        h_ad_cost_r[d] = d_ad_cost_memory + (d * img_sz + cost_sz);
    }
    
    checkCudaError(cudaMemcpy(d_ad_cost_l, h_ad_cost_l, sizeof(float*) * num_disp, cudaMemcpyHostToDevice));
    checkCudaError(cudaMemcpy(d_ad_cost_r, h_ad_cost_r, sizeof(float*) * num_disp, cudaMemcpyHostToDevice));
    
    // Launch Kernel
    ci_ad_kernel<<<grid_sz, block_sz>>>(d_img_l, d_img_r, d_ad_cost_l, num_disp, zero_disp, 1, num_rows, num_cols, elem_sz );
    ci_ad_kernel<<<grid_sz, block_sz>>>(d_img_r, d_img_l, d_ad_cost_r, num_disp, zero_disp, -1, num_rows, num_cols, elem_sz );
    cudaDeviceSynchronize();

    ////////////
    // CENSUS //
    ////////////

    // Device Memory Allocation & Copy
    unsigned long long* d_census_l;
    unsigned long long* d_census_r;

    checkCudaError(cudaMalloc(&d_census_l, sizeof(unsigned long long) * imgelem_sz));
    checkCudaError(cudaMalloc(&d_census_r, sizeof(unsigned long long) * imgelem_sz));

    // Launch Census Transform Kernel
    tx_census_9x7_kernel<<<grid_sz, block_sz>>>(d_img_l, d_census_l, num_rows, num_cols, elem_sz);
    tx_census_9x7_kernel<<<grid_sz, block_sz>>>(d_img_r, d_census_r, num_rows, num_cols, elem_sz);
    cudaDeviceSynchronize();
    
    // Cost Initialization Device Cost Memory
    float** d_census_cost_l;
    float** d_census_cost_r;
    float* d_census_cost_memory;
    
    checkCudaError(cudaMalloc(&d_census_cost_l, sizeof(float*) * num_disp));
    checkCudaError(cudaMalloc(&d_census_cost_r, sizeof(float*) * num_disp));
    
    float** h_census_cost_l = (float**) malloc(sizeof(float*) * num_disp);
    float** h_census_cost_r = (float**) malloc(sizeof(float*) * num_disp);

    checkCudaError(cudaMalloc(&d_census_cost_memory, sizeof(float) * cost_sz * 2));
    
    for (int d = 0; d < num_disp; ++d)
    {
        h_census_cost_l[d] = d_census_cost_memory + (d * img_sz);
        h_census_cost_r[d] = d_census_cost_memory + (d * img_sz + cost_sz);
    }
    
    checkCudaError(cudaMemcpy(d_census_cost_l, h_census_cost_l, sizeof(float*) * num_disp, cudaMemcpyHostToDevice));
    checkCudaError(cudaMemcpy(d_census_cost_r, h_census_cost_r, sizeof(float*) * num_disp, cudaMemcpyHostToDevice));
    
    // Launch Kernel
    ci_census_kernel<<<grid_sz, block_sz>>>(d_census_l, d_census_r, d_census_cost_l, 
                                            num_disp, zero_disp, 1, num_rows, num_cols, elem_sz);
    ci_census_kernel<<<grid_sz, block_sz>>>(d_census_r, d_census_l, d_census_cost_r, 
                                            num_disp, zero_disp, -1, num_rows, num_cols, elem_sz);
    cudaDeviceSynchronize();

    /////////////////
    // AD + CENSUS //
    /////////////////

    for (int d = 0; d < num_disp; ++d)
    {
        h_adcensus_cost_l[d] = d_adcensus_cost_memory + (d * img_sz);
        h_adcensus_cost_r[d] = d_adcensus_cost_memory + (d * img_sz + cost_sz);
    }
    
    checkCudaError(cudaMemcpy(d_adcensus_cost_l, h_adcensus_cost_l, sizeof(float*) * num_disp, cudaMemcpyHostToDevice));
    checkCudaError(cudaMemcpy(d_adcensus_cost_r, h_adcensus_cost_r, sizeof(float*) * num_disp, cudaMemcpyHostToDevice));
    
    // Launch Kernel
    ci_adcensus_kernel<<<grid_sz, block_sz>>>(d_ad_cost_l, d_ad_cost_r, d_census_cost_l, d_census_cost_r, d_adcensus_cost_l, d_adcensus_cost_r, 1.0/ad_coeff, 1.0/census_coeff, num_disp, zero_disp, num_rows, num_cols, elem_sz);
    cudaDeviceSynchronize();

    /////////
    // END //
    /////////
    
    // Device De-allocation
    cudaFree(d_ad_cost_memory);
    cudaFree(d_ad_cost_l);
    cudaFree(d_ad_cost_r);
    
    cudaFree(d_census_l);
    cudaFree(d_census_r);
    cudaFree(d_census_cost_memory);
    cudaFree(d_census_cost_l);
    cudaFree(d_census_cost_r);

    // Host De-allocation
    free(h_ad_cost_l); 
    free(h_ad_cost_r); 
    free(h_census_cost_l); 
    free(h_census_cost_r); 
}

void ci_adcensus(unsigned char* img_l, unsigned char* img_r, float** cost_l, float** cost_r, 
                 float ad_coeff, float census_coeff, int num_disp, int zero_disp, 
                 int num_rows, int num_cols, int elem_sz)
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
    
    //
    startCudaTimer(&timer);
    ci_ad_kernel_2<<<grid_sz, block_sz>>>(d_img_l, d_img_r, d_ad_cost_l, d_ad_cost_r, num_disp, zero_disp, num_rows, num_cols, elem_sz);
    stopCudaTimer(&timer, "Absolute Difference Kernel #2");
    // Launch Kernel
    startCudaTimer(&timer);
    ci_ad_kernel_3<<<grid_sz, block_sz>>>(d_img_l, d_img_r, d_ad_cost_l, d_ad_cost_r, num_disp, zero_disp, num_rows, num_cols, elem_sz);
    stopCudaTimer(&timer, "Absolute Difference Kernel #3");

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
    
    // Launch Kernel
    startCudaTimer(&timer);
    ci_census_kernel<<<grid_sz, block_sz>>>(d_census_l, d_census_r, d_census_cost_l, num_disp, zero_disp, 1, num_rows, num_cols, elem_sz);
    stopCudaTimer(&timer, "Census Cost Kernel");
    
    startCudaTimer(&timer);
    ci_census_kernel<<<grid_sz, block_sz>>>(d_census_r, d_census_l, d_census_cost_r, num_disp, zero_disp, -1, num_rows, num_cols, elem_sz);
    stopCudaTimer(&timer, "Census Cost Kernel");
    
    /////////////////
    // AD + CENSUS //
    /////////////////
    
    float** d_adcensus_cost_l;
    float** d_adcensus_cost_r;
    
    checkCudaError(cudaMalloc(&d_adcensus_cost_l, sizeof(float*) * num_disp));
    checkCudaError(cudaMalloc(&d_adcensus_cost_r, sizeof(float*) * num_disp));
    
    float** h_adcensus_cost_l = (float**) malloc(sizeof(float*) * num_disp);
    float** h_adcensus_cost_r = (float**) malloc(sizeof(float*) * num_disp);
    
    for (int d = 0; d < num_disp; ++d)
    {
        checkCudaError(cudaMalloc(&h_adcensus_cost_l[d], sizeof(float) * num_rows * num_cols));
        checkCudaError(cudaMalloc(&h_adcensus_cost_r[d], sizeof(float) * num_rows * num_cols));
    }
    
    checkCudaError(cudaMemcpy(d_adcensus_cost_l, h_adcensus_cost_l, sizeof(float*) * num_disp, cudaMemcpyHostToDevice));
    checkCudaError(cudaMemcpy(d_adcensus_cost_r, h_adcensus_cost_r, sizeof(float*) * num_disp, cudaMemcpyHostToDevice));
    
    // Launch Kernel
    startCudaTimer(&timer);
    ci_adcensus_kernel<<<grid_sz, block_sz>>>(d_ad_cost_l, d_ad_cost_r, d_census_cost_l, d_census_cost_r, d_adcensus_cost_l, d_adcensus_cost_r, 1.0/ad_coeff, 1.0/census_coeff, num_disp, zero_disp, num_rows, num_cols, elem_sz);
    stopCudaTimer(&timer, "Ad + Census Cost Kernel");

    // Copy Memory Device -> Host
    for (int d = 0; d < num_disp; ++d)
    {
        checkCudaError(cudaMemcpy(cost_l[d], h_adcensus_cost_l[d], sizeof(float) * num_rows * num_cols, cudaMemcpyDeviceToHost));
        checkCudaError(cudaMemcpy(cost_r[d], h_adcensus_cost_r[d], sizeof(float) * num_rows * num_cols, cudaMemcpyDeviceToHost));
    }

    /////////
    // END //
    /////////
    
    // Device De-allocation
    cudaFree(d_img_l);
    cudaFree(d_img_r);
    cudaFree(d_census_l);
    cudaFree(d_census_r);
    cudaFree(d_ad_cost_l);
    cudaFree(d_ad_cost_r);
    cudaFree(d_census_cost_l);
    cudaFree(d_census_cost_r);
    cudaFree(d_adcensus_cost_l);
    cudaFree(d_adcensus_cost_r);
    for (int d = 0; d < num_disp; ++d)
    {
        cudaFree(h_ad_cost_l[d]);
        cudaFree(h_ad_cost_r[d]);
        cudaFree(h_census_cost_l[d]);
        cudaFree(h_census_cost_r[d]);
        cudaFree(h_adcensus_cost_l[d]);
        cudaFree(h_adcensus_cost_r[d]);
    }

    // Host De-allocation
    free(h_ad_cost_l); 
    free(h_ad_cost_r); 
    free(h_census_cost_l); 
    free(h_census_cost_r); 
    free(h_adcensus_cost_l); 
    free(h_adcensus_cost_r); 
}

#endif
