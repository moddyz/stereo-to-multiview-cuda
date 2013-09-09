#ifndef D_FILTER_BILATERAL_KERNEL
#define D_FILTER_BILATERAL_KERNEL
#include "d_filter_bilateral.h"

inline __device__ float gaussian1D(float x, float sigma)
{
    float variance = pow(sigma, 2);
    float power = pow(x, 2);
    float exponent = -power/(2*variance);
    return __expf(exponent) / sqrt(2 * PI * variance);
}

__global__ void filter_bilateral_1_kernel(float *img_out, float *img_in, float* kernel,
                                          int radius, float sigma_color, float sigma_spatial,
                                          int num_rows, int num_cols)
{
    int tx = threadIdx.x + blockIdx.x * blockDim.x;
    int ty = threadIdx.y + blockIdx.y * blockDim.y;

    if ((tx > num_cols - 1) || (ty > num_rows - 1))
        return;

    float val_a = img_in[tx + ty * num_cols];

    int kernel_width = radius * 2 + 1;
    float norm = 0.0f;
    float res = 0.0f;

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

            float val_s = img_in[sx + sy * num_cols];

            float val_gspatial = kernel[(x + radius) + (y + radius) * kernel_width];
            float val_gcolor = gaussian1D(val_a - val_s, sigma_color);
            float weight = val_gspatial * val_gcolor;
            
            norm = norm + weight;
            res = res + (val_s * weight); 
        }
    }

    res /= norm;

    img_out[tx + ty * num_cols] = res;
}


void d_filter_bilateral_1(float *d_img,
                          int radius, float sigma_color, float sigma_spatial,
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
    float* kernel = (float*) malloc(sizeof(float) * kernel_sz);
    generateGaussianKernel(kernel, radius, sigma_spatial);
    
    // Device Memory Allocation & Copy
    float* d_img_out;
    checkCudaError(cudaMalloc(&d_img_out, sizeof(float) * num_rows * num_cols));
    
    float* d_kernel;
    checkCudaError(cudaMalloc(&d_kernel, sizeof(float) * kernel_sz));
    checkCudaError(cudaMemcpy(d_kernel, kernel, sizeof(float) * kernel_sz, cudaMemcpyHostToDevice));
    
    filter_bilateral_1_kernel<<<grid_sz, block_sz>>>(d_img_out, d_img, d_kernel, radius, sigma_color, sigma_spatial, num_rows, num_cols);

    checkCudaError(cudaMemcpy(d_img, d_img_out, sizeof(float) * num_rows * num_cols, cudaMemcpyDeviceToDevice));
    
    cudaFree(d_img_out);
    free(kernel);
    cudaFree(d_kernel);
}

void filter_bilateral_1(float *img,
                        int radius, float sigma_color, float sigma_spatial,
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
    filter_bilateral_1_kernel<<<grid_sz, block_sz>>>(d_img_out, d_img_in, d_kernel, radius, sigma_color, sigma_spatial, num_rows, num_cols);
    stopCudaTimer(&timer, "Bilateral Filter (1 Component) Kernel");
    
    checkCudaError(cudaMemcpy(img, d_img_out, sizeof(float) * num_rows * num_cols, cudaMemcpyDeviceToHost));

    free(kernel);
    cudaFree(d_kernel);
    cudaFree(d_img_out);
    cudaFree(d_img_in);
}

#endif
