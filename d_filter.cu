#ifndef D_FILTER_KERNEL
#define D_FILTER_KERNEL
#include "d_filter.h"
#include "cuda_utils.h"
#include <math.h>

__global__ void filter_median_kernel(float *out, float *in, 
                                     int num_rows, int num_cols)
{
    int tx = threadIdx.x + blockIdx.x * blockDim.x;
    int ty = threadIdx.y + blockIdx.y * blockDim.y;

    if ((tx > num_cols - 1) || (ty > num_rows - 1))
        return;

    float values[9];
    int count = 0; 
    for (int y = -1; y <= 1; ++y)
    {
        for (int x = -1; x <= 1; ++x)
        {
            int sx = tx + x;
            int sy = ty + y;

            values[count] = in[sx + sy * num_cols];
            ++count;
        }
    }
    
    for (int i = 0; i < 9; ++i)
    {
        int cur = values[i];
        for (int j = i; j < 9; ++j)
        {
            int comp = values[j];
            if (comp < cur)
            {
                values[j] = cur;
                values[i] = comp;
                cur = comp;
            }
        }
    }
    out[tx + ty * num_cols] = values[4];
}

void d_filter_median(float *d_img_in,
                   int num_rows, int num_cols)
{
    cudaEventPair_t timer;

    // Setup Block & Grid Size
    size_t bw = 32;
    size_t bh = 32;
    
    size_t gw = (num_cols + bw - 1) / bw;
    size_t gh = (num_rows + bh - 1) / bh;
    
    const dim3 block_sz(bw, bh, 1);
    const dim3 grid_sz(gw, gh, 1);

    float *d_img_out;
    checkCudaError(cudaMalloc(&d_img_out, sizeof(float) * num_rows * num_cols));
    
    filter_median_kernel<<<grid_sz, block_sz>>>(d_img_out, d_img_in, num_rows, num_cols);
    
    checkCudaError(cudaMemcpy(d_img_in, d_img_out, sizeof(float) * num_rows * num_cols, cudaMemcpyDeviceToDevice));

    cudaFree(d_img_out);
}

void filter_median(float *img,
                   int num_rows, int num_cols)
{
    cudaEventPair_t timer;

    // Setup Block & Grid Size
    size_t bw = 32;
    size_t bh = 32;
    
    size_t gw = (num_cols + bw - 1) / bw;
    size_t gh = (num_rows + bh - 1) / bh;
    
    const dim3 block_sz(bw, bh, 1);
    const dim3 grid_sz(gw, gh, 1);

    float *d_img_in;
    float *d_img_out;

    checkCudaError(cudaMalloc(&d_img_in, sizeof(float) * num_rows * num_cols));
    checkCudaError(cudaMemcpy(d_img_in, img, sizeof(float) * num_rows * num_cols, cudaMemcpyHostToDevice));

    checkCudaError(cudaMalloc(&d_img_out, sizeof(float) * num_rows * num_cols));
    
    startCudaTimer(&timer);
    filter_median_kernel<<<grid_sz, block_sz>>>(d_img_out, d_img_in, num_rows, num_cols);
    stopCudaTimer(&timer, "Median Filter (1 Component) Kernel");

    checkCudaError(cudaMemcpy(img, d_img_out, sizeof(float) * num_rows * num_cols, cudaMemcpyDeviceToHost));

    cudaFree(d_img_in);
    cudaFree(d_img_out);
}
                                       
__global__ void filter_bleed_1_kernel(unsigned char *img_out, unsigned char *img_in,
                                      int radius, int kernel_sz,
                                      int num_rows, int num_cols)
{
    int tx = threadIdx.x + blockIdx.x * blockDim.x;
    int ty = threadIdx.y + blockIdx.y * blockDim.y;

    if ((tx > num_cols - 1) || (ty > num_rows - 1))
        return;
    
    unsigned char val_a = img_in[tx + ty * num_cols];
    int val_count = 0; 
    for (int y = -radius; y <= radius; ++y)
    {
        for (int x = -radius; x <= radius; ++x)
        {
            int sx = tx + x;
            int sy = ty + y;

            if (sx < 0) sx = -sx;
            if (sy < 0) sy = -sy;
            if (sx > num_cols - 1) sx = num_cols - 1 - x;
            if (sy > num_rows - 1) sy = num_rows - 1 - y;

            unsigned char val_s = img_in[sx + sy * num_cols];
            if (val_s > 0)
                val_count = val_count + 1;
        }
    }

    if (val_count > (kernel_sz - 1) * 0.30)
        img_out[tx + ty * num_cols] = 1;
    else
        img_out[tx + ty * num_cols] = val_a;
}


void d_filter_bleed_1(unsigned char *d_img,
                    int radius,
                    int num_rows, int num_cols)
{
    // Setup Block & Grid Size
    size_t bw = 32;
    size_t bh = 32;
    
    size_t gw = (num_cols + bw - 1) / bw;
    size_t gh = (num_rows + bh - 1) / bh;
    
    const dim3 block_sz(bw, bh, 1);
    const dim3 grid_sz(gw, gh, 1);

    int kernel_sz = (2 * radius + 1) * (2 * radius + 1);
    
    unsigned char* d_img_out;
    checkCudaError(cudaMalloc(&d_img_out, sizeof(unsigned char) * num_rows * num_cols));
    
    filter_bleed_1_kernel<<<grid_sz, block_sz>>>(d_img_out, d_img, radius, kernel_sz, num_rows, num_cols);
    cudaDeviceSynchronize();

    checkCudaError(cudaMemcpy(d_img, d_img_out, sizeof(unsigned char) * num_rows * num_cols, cudaMemcpyDeviceToDevice));

    cudaFree(d_img_out);
}

void filter_bleed_1(unsigned char *img,
                    int radius,
                    int num_rows, int num_cols)
{
    cudaEventPair_t timer;

    // Setup Block & Grid Size
    size_t bw = 32;
    size_t bh = 32;
    
    size_t gw = (num_cols + bw - 1) / bw;
    size_t gh = (num_rows + bh - 1) / bh;
    
    const dim3 block_sz(bw, bh, 1);
    const dim3 grid_sz(gw, gh, 1);

    int kernel_sz = (2 * radius + 1) * (2 * radius + 1);

    unsigned char* d_img_in;
    unsigned char* d_img_out;

    checkCudaError(cudaMalloc(&d_img_in, sizeof(unsigned char) * num_rows * num_cols));
    checkCudaError(cudaMemcpy(d_img_in, img, sizeof(unsigned char) * num_rows * num_cols, cudaMemcpyHostToDevice));

    checkCudaError(cudaMalloc(&d_img_out, sizeof(unsigned char) * num_rows * num_cols));
    
    startCudaTimer(&timer);
    filter_bleed_1_kernel<<<grid_sz, block_sz>>>(d_img_out, d_img_in, radius, kernel_sz, num_rows, num_cols);
    stopCudaTimer(&timer, "Bleed Filter (1 Component) Kernel");

    checkCudaError(cudaMemcpy(img, d_img_out, sizeof(unsigned char) * num_rows * num_cols, cudaMemcpyDeviceToHost));

    cudaFree(d_img_in);
    cudaFree(d_img_out);
}


#endif
