#ifndef D_FILTER_GAUSSIAN_KERNEL
#define D_FILTER_GAUSSIAN_KERNEL
#include "d_filter_gaussian.h"
#include "cuda_utils.h"
#include <math.h>

#define PI 3.14159265359f

__global__ void filter_gaussian_1_kernel_1(float* img_out, float* img_in,
                                           float *kernel,
                                           int radius, float sigma_spatial,
                                           int num_rows, int num_cols,
                                           int sm_img_rows, int sm_img_cols, int sm_img_sz, int sm_img_padding,
                                           int sm_kernel_len, int sm_kernel_sz)
{
    int gx = threadIdx.x + blockIdx.x * blockDim.x;
    int gy = threadIdx.y + blockIdx.y * blockDim.y;
    
    if ((gx > num_cols - 1) || (gy > num_rows - 1))
        return;

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    extern __shared__ float sm_memory[];
    float* sm_img = sm_memory;
    float* sm_kernel = sm_memory + sm_img_sz;
    
    // Populate Shared Memory IMG
    for (int gsy = gy - sm_img_padding, tsy = ty;
         tsy < sm_img_rows;
         gsy += blockDim.y, tsy += blockDim.y)
    {
         for (int gsx = gx - sm_img_padding, tsx = tx; 
              tsx < sm_img_cols;
              gsx += blockDim.x, tsx += blockDim.x)
         {
             int sm_idx = tsx + tsy * sm_img_cols;
             int gm_idx = min(max(gsx, 0), num_cols - 1) + min(max(gsy, 0), num_rows - 1) * num_cols;

             sm_img[sm_idx] = img_in[gm_idx];
         }
    }

    for (int tsy = ty;
         tsy < sm_kernel_len;
         tsy += blockDim.y)
    {
         for (int tsx = tx; 
              tsx < sm_kernel_len;
              tsx += blockDim.x)
         {
             int idx = tsx + tsy * sm_kernel_len;

             sm_kernel[idx] = kernel[idx];
         }
    }

    __syncthreads();

    
    float val_a = sm_img[tx + sm_img_padding + (ty + sm_img_padding) * sm_img_cols];
    
    int kernel_width = radius * 2 + 1;
    float res = 0.0f;
    float norm = 0.0f;

    for (int y = -radius; y <= radius; ++y)
    {
        int sy = ty + sm_img_padding + y;
        int sy_sm_img_cols = sy * sm_img_cols;
        int y_radius_kernel_width = (y + radius) * kernel_width;
        for (int x = -radius; x <= radius; ++x)
        {
            int sx = tx + sm_img_padding + x;

            float val_s = sm_img[sx + sy_sm_img_cols];
            float weight = sm_kernel[x + radius + y_radius_kernel_width];
            
            norm = norm + weight;
            res = res + (val_s * weight); 
        }
    }
    if (val_a < res/norm)
        img_out[gx + gy * num_cols] = res/norm;
    else
        img_out[gx + gy * num_cols] = val_a;
}



__global__ void filter_gaussian_1_kernel(float* img_out, float* img_in,
                                          float *kernel,
                                          int radius, float sigma_spatial,
                                          int num_rows, int num_cols)
{
    int gx = threadIdx.x + blockIdx.x * blockDim.x;
    int gy = threadIdx.y + blockIdx.y * blockDim.y;
    
    if ((gx > num_cols - 1) || (gy > num_rows - 1))
        return;
    
    float val_a = img_in[gx + gy * num_cols];
    int kernel_width = radius * 2 + 1;
    float res = 0.0f;
    float norm = 0.0f;

    for (int y = -radius; y <= radius; ++y)
    {
        for (int x = -radius; x <= radius; ++x)
        {
            int sx = gx + x;
            int sy = gy + y;

            if (sx < 0) sx = -sx;
            if (sy < 0) sy = -sy;
            if (sx > num_cols - 1) sx = num_cols - 1 - x;
            if (sy > num_rows - 1) sy = num_rows - 1 - y;

            float val_s = img_in[sx + sy * num_cols];
            float weight = kernel[(x + radius) + (y + radius) * kernel_width];
            
            norm = norm + weight;
            res = res + (val_s * weight); 
            
        }
    }
    if (val_a < res/norm)
        img_out[gx + gy * num_cols] = res/norm;
    else
        img_out[gx + gy * num_cols] = val_a;
}

void d_filter_gaussian_1(float *d_img,
                          int radius, float sigma_spatial,
                          int num_rows, int num_cols)
{
    // Setup Block & Grid Size
    size_t bw = 32;
    size_t bh = 32;
    
    size_t gw = (num_cols + bw - 1) / bw;
    size_t gh = (num_rows + bh - 1) / bh;
    
    const dim3 block_sz(bw, bh, 1);
    const dim3 grid_sz(gw, gh, 1);

    int sm_img_rows = bh + 2 * radius;
    int sm_img_cols = bw + 2 * radius;
    int sm_img_sz = sm_img_rows * sm_img_cols;
    int sm_img_padding = radius;

    int sm_kernel_len = 2 * radius + 1;
    int sm_kernel_sz = sm_kernel_len * sm_kernel_len; 
    
    int kernel_sz = sm_kernel_sz; 
    float* kernel = (float*) malloc(sizeof(float) * kernel_sz);
    generateGaussianKernel(kernel, radius, sigma_spatial);
    
    // Device Memory Allocation & Copy
    float* d_img_out;

    checkCudaError(cudaMalloc(&d_img_out, sizeof(float) * num_rows * num_cols));

    float* d_kernel;
    checkCudaError(cudaMalloc(&d_kernel, sizeof(float) * kernel_sz));
    checkCudaError(cudaMemcpy(d_kernel, kernel, sizeof(float) * kernel_sz, cudaMemcpyHostToDevice));
    
    filter_gaussian_1_kernel_1<<<grid_sz, block_sz, sizeof(float) * (sm_img_sz + sm_kernel_sz)>>>(d_img_out, d_img, d_kernel, radius, sigma_spatial, num_rows, num_cols, sm_img_rows, sm_img_cols, sm_img_sz, sm_img_padding, sm_kernel_len, sm_kernel_sz);
    cudaDeviceSynchronize();
    
    checkCudaError(cudaMemcpy(d_img, d_img_out, sizeof(float) * num_rows * num_cols, cudaMemcpyDeviceToDevice));

    free(kernel);
    cudaFree(d_kernel);
    cudaFree(d_img_out);
}


void filter_gaussian_1(float *img,
                        int radius, float sigma_spatial,
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

    int sm_img_rows = bh + 2 * radius;
    int sm_img_cols = bw + 2 * radius;
    int sm_img_sz = sm_img_rows * sm_img_cols;
    int sm_img_padding = radius;

    int sm_kernel_len = 2 * radius + 1;
    int sm_kernel_sz = sm_kernel_len * sm_kernel_len; 
    
    int kernel_sz = sm_kernel_sz; 
    float* kernel = (float*) malloc(sizeof(float) * kernel_sz);
    generateGaussianKernel(kernel, radius, sigma_spatial);
    
    // Device Memory Allocation & Copy
    float* d_img_in;
    float* d_img_out;

    checkCudaError(cudaMalloc(&d_img_in, sizeof(float) * num_rows * num_cols));
    checkCudaError(cudaMemcpy(d_img_in, img, sizeof(float) * num_rows * num_cols, cudaMemcpyHostToDevice));

    checkCudaError(cudaMalloc(&d_img_out, sizeof(float) * num_rows * num_cols));

    float* d_kernel;
    checkCudaError(cudaMalloc(&d_kernel, sizeof(float) * kernel_sz));
    checkCudaError(cudaMemcpy(d_kernel, kernel, sizeof(float) * kernel_sz, cudaMemcpyHostToDevice));
    
    startCudaTimer(&timer);
    filter_gaussian_1_kernel<<<grid_sz, block_sz>>>(d_img_out, d_img_in, d_kernel, radius, sigma_spatial, num_rows, num_cols);
    stopCudaTimer(&timer, "Gaussian Filter (1 FLoat Component) Kernel");
    
    startCudaTimer(&timer);
    filter_gaussian_1_kernel_1<<<grid_sz, block_sz, sizeof(float) * (sm_img_sz + sm_kernel_sz)>>>(d_img_out, d_img_in, d_kernel, radius, sigma_spatial, num_rows, num_cols, sm_img_rows, sm_img_cols, sm_img_sz, sm_img_padding, sm_kernel_len, sm_kernel_sz);
    stopCudaTimer(&timer, "Gaussian Filter (1 FLoat Component) Kernel");
    
    checkCudaError(cudaMemcpy(img, d_img_out, sizeof(float) * num_rows * num_cols, cudaMemcpyDeviceToHost));

    free(kernel);
    cudaFree(d_kernel);
    cudaFree(d_img_out);
    cudaFree(d_img_in);
}

__host__ __device__ float gaussian2D(float x, float y, float sigma)
{
    float variance = pow(sigma,2);
    float exponent = -(pow(x,2) + pow(y,2))/(2 * variance);
    return expf(exponent) / (2 * PI * variance);
}

void generateGaussianKernel(float* kernel, int radius, float sigma)
{
    int kernel_width = radius * 2 + 1;

    for (int y = -radius; y <= radius; ++y)
    {
        for (int x = -radius; x <= radius; ++x)
        {
            kernel[(x + radius) + (y + radius) * kernel_width] = gaussian2D(x, y, sigma);
        }
    }
}

#endif
