#ifndef functkernel
#define functkernel
#include <limits.h>
#include <float.h>
#include <stdio.h>
#include "function.h"

__global__ void exc_scan(unsigned int* input, unsigned int* out, int sz)
{
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    if (id >= sz)
        return;

    unsigned int acc = input[id];

    for (int c = 1; c <= sz; c = c*2)
    {
        if (id >= c)
        {
            acc = acc + input[id - c] + out[id - c];
            __syncthreads();
            out[id] = acc;
            __syncthreads();
        }
    }
    if (id == 0)
        out[id] = input[id];
}

__global__ void make_histogram(unsigned int* hist, const float* const values, float min, float range, int numBins, int sz)
{
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    if (id >= sz)
        return;
    int bin = (values[id] - min) / range * numBins;
    if (bin >= numBins)
        bin--;
    atomicAdd(&(hist[bin]), 1);
}

__global__ void reduce_min( float *in, float *min )
{
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    int sz = gridDim.x * blockDim.x;

    for (int s = sz / 2; s > 0; s >>=1)
    {
        if (id < s)
        {
            in[id] = fmin(in[id], in[id + s]);
        }
        __syncthreads();
    }

    if (id == 0)
        *min = in[id];
}

__global__ void reduce_max( float *in, float *max )
{
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    int sz = gridDim.x * blockDim.x;

    for (int s = sz / 2; s > 0; s >>=1)
    {
        if (id < s)
        {
            in[id] = fmax(in[id], in[id + s]);
        }
        __syncthreads();
    }

    if (id == 0)
        *max = in[id];
}

// Globals
float *d_min;
float *d_max;
unsigned int *d_histogram;
float *d_temp;
float *d_temp2;

void d_alloc(const size_t num_rows, const size_t num_cols, const size_t num_bins)
{
    cudaMalloc(&d_min, sizeof(float)); 
    cudaMalloc(&d_max, sizeof(float));
    cudaMalloc(&d_histogram, sizeof(unsigned int) * num_bins);
    cudaMemset(d_histogram, 0, sizeof(unsigned int) * num_bins);
}

void d_dealloc()
{
    cudaFree(d_min);
    cudaFree(d_max);
    cudaFree(d_temp);
    cudaFree(d_temp2);
    cudaFree(d_histogram);
}

void histogram_process(float *h_img, float &h_min, float &h_max,  unsigned int *h_cdf, size_t num_cols, size_t num_rows, size_t num_bins)
{
    d_alloc(num_rows, num_cols, num_bins);

    float *d_img;
    cudaMalloc(&d_img, sizeof(float) * num_rows * num_cols);
    cudaMemcpy(d_img, h_img, sizeof(float) * num_rows * num_cols, cudaMemcpyHostToDevice);

    // Kernel Size
    size_t bw = 1024;
    size_t sz = num_rows * num_cols;
    size_t gw = (sz + bw - 1) / bw;
    printf("gw: %d\n", (int)gw);

    const dim3 bs(bw, 1, 1);
    const dim3 gs(gw, 1, 1);
    
    // Min REDUCE
    cudaMalloc(&d_temp, sizeof(float) * bw * gw);
    
    float *h_temp = (float*) malloc(sizeof(float) * bw * gw);
    for (int i = 0; i < bw * gw; ++i)
        h_temp[i] = FLT_MAX;
   
    cudaMemcpy(d_temp, h_temp, sizeof(float) * bw * gw, cudaMemcpyHostToDevice);
    cudaMemcpy(d_temp, d_img, sizeof(float) * num_rows * num_cols, cudaMemcpyDeviceToDevice);
    
    reduce_min<<<gs, bs>>>(d_temp, d_min);
    cudaMemcpy(&h_min, d_min, sizeof(float), cudaMemcpyDeviceToHost);
    
    // Max REDUCE
    for (int i = 0; i < bw * gw; ++i)
        h_temp[i] = h_min;
    
    cudaMalloc(&d_temp2, sizeof(float) * bw * gw);
    cudaMemcpy(d_temp2, h_temp, sizeof(float) * bw * gw, cudaMemcpyHostToDevice);
    cudaMemcpy(d_temp2, d_img, sizeof(float) * num_rows * num_cols, cudaMemcpyDeviceToDevice);
    
    reduce_max<<<gs, bs>>>(d_temp2, d_max);
    cudaMemcpy(&h_max, d_max, sizeof(float), cudaMemcpyDeviceToHost);
    
    // Histogram
    make_histogram<<<gs,bs>>>(d_histogram, d_img, h_min, h_max - h_min, num_bins, num_rows * num_cols);
    cudaDeviceSynchronize();

    cudaMemcpy(h_cdf, d_histogram, sizeof(unsigned int) * num_bins, cudaMemcpyDeviceToHost);
    printf("Histogram:\n");
    for (int i = 0; i < num_bins; ++i)
        printf("%d ", (int) h_cdf[i]);
    printf("\n");

    // Exclusive Prefix Sum
    bw = 5;
    sz = num_bins;
    gw = (sz + bw - 1) / bw;
    const dim3 bs2(bw, 1, 1);
    const dim3 gs2(gw, 1, 1);

    unsigned int *d_temp3;
    cudaMalloc(&d_temp3, sizeof(unsigned int) * num_bins);
    cudaMemset(d_temp3, 0, sizeof(unsigned int) * num_bins);
    exc_scan<<<gs, bs>>>(d_histogram, d_temp3, num_bins);
    
    cudaMemcpy(h_cdf, d_temp3, sizeof(unsigned int) * num_bins, cudaMemcpyDeviceToHost);
    printf("Prefix Summed Histogram:\n");
    for (int i = 0; i < num_bins; ++i)
        printf("%d ", (int) h_cdf[i]);
    printf("\n");

    free(h_temp);
    cudaFree(d_temp3);
    d_dealloc();
}


#endif
