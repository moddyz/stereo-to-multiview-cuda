#ifndef D_MUX_MULTIVIEW_KERNEL
#define D_MUX_MULTIVIEW_KERNEL
#include "d_mux_multiview.h"
#include "cuda_utils.h"
#include "d_alu.h"
#include <math.h>

#define PI 3.1415926535f

inline __device__ unsigned char fast_bilinear_interp(unsigned char* data, int elem_sz, int elem_offset, float coord_x, float coord_y, int width, int height) 
{
    int coord_00_x = floor(coord_x);
    int coord_00_y = floor(coord_y);
    
    int coord_01_x = min(coord_00_x + 1, width - 1);
    int coord_01_y = coord_00_y;

    int coord_10_x = coord_00_x;
    int coord_10_y = min(coord_00_y + 1, height - 1);
    
    int coord_11_x = min(coord_00_x + 1, width - 1);
    int coord_11_y = min(coord_00_y + 1, height - 1);

    float weight_x = coord_x - (float) coord_00_x;
    float weight_y = coord_y - (float) coord_00_y;

    unsigned char val_00 = data[(coord_00_x + coord_00_y * width) * elem_sz + elem_offset];
    unsigned char val_01 = data[(coord_01_x + coord_01_y * width) * elem_sz + elem_offset];
    unsigned char val_10 = data[(coord_10_x + coord_10_y * width) * elem_sz + elem_offset];
    unsigned char val_11 = data[(coord_11_x + coord_11_y * width) * elem_sz + elem_offset];

    float top = (float) val_00 * (1.0f - weight_x) + (float) val_01 * weight_x;
    float bot = (float) val_10 * (1.0f - weight_x) + (float) val_11 * weight_x;
    
    return (unsigned char) (top * (1.0f - weight_y) + bot * weight_y);
}

__global__ void mux_multiview_kernel_2(unsigned char** views, unsigned char* output, 
                                       int num_views, float y_interval, float inv_y_interval,
									   int num_rows_in, int num_cols_in, int num_rows_out, int num_cols_out, int elem_sz)
{
    // Thread Id's
    int gx = num_views * threadIdx.x + blockIdx.x * blockDim.x;
    int gy = threadIdx.y + blockIdx.y * blockDim.y;

    if (gx >= num_cols_out || gy >= num_rows_out)
        return;

    // Compute Input Sampling Coordinates
    for (int v = 0; v < num_views; ++v)
    {
        int tx = gx + v;
        int ty = gy;

        float x_samp = fmin(fmax(((float) tx / (float) num_cols_out) * (float) num_cols_in, 0), (float) (num_cols_in - 1));
        float y_samp = fmin(fmax(((float) ty / (float) num_rows_out) * (float) num_rows_in, 0), (float) (num_rows_in - 1));
        
        // Interlace Specific
        float x_interval = num_views;
        float y_view = ty % ((int) round(y_interval)) + 1.0f;
        y_view = y_view * x_interval * inv_y_interval;
        int x_view = (tx * 3 + (int) y_view) % ((int) x_interval);
        int r_view = x_view;
        if (r_view < 0)
            r_view = r_view + num_views;
        int g_view = r_view + 1;
        int b_view = r_view + 2;
        if (g_view >= num_views)
            g_view = g_view - num_views;
        if (b_view >= num_views)
            b_view =  b_view - num_views;
        
        // Write to Output
        int b_out = (tx + ty * num_cols_out) * elem_sz;
        int g_out = b_out + 1;
        int r_out = g_out + 1;

        output[b_out] = fast_bilinear_interp(views[b_view], elem_sz, 0, x_samp, y_samp, num_cols_in, num_rows_in);
        output[g_out] = fast_bilinear_interp(views[g_view], elem_sz, 1, x_samp, y_samp, num_cols_in, num_rows_in);
        output[r_out] = fast_bilinear_interp(views[r_view], elem_sz, 2, x_samp, y_samp, num_cols_in, num_rows_in);
    }
}

__global__ void mux_multiview_kernel(unsigned char** views, unsigned char* output, 
                                     int num_views, float y_interval,
									 int num_rows_in, int num_cols_in, int num_rows_out, int num_cols_out, int elem_sz)
{
    // Thread Id's
    int tx = threadIdx.x + blockIdx.x * blockDim.x;
    int ty = threadIdx.y + blockIdx.y * blockDim.y;

    if (tx >= num_cols_out || ty >= num_rows_out)
        return;
    
    // Compute Input Sampling Coordinates
    float x_samp = fmin(fmax(((float) tx / (float) num_cols_out) * (float) num_cols_in, 0), (float) (num_cols_in - 1));
    float y_samp = fmin(fmax(((float) ty / (float) num_rows_out) * (float) num_rows_in, 0), (float) (num_rows_in - 1));
    
    // Interlace Specific
    float x_interval = num_views;
    float y_view = ty % ((int) round(y_interval)) + 1.0;
    y_view = y_view * x_interval / y_interval;
    int x_view = (tx * 3 + (int) y_view) % ((int) x_interval);
    int r_view = x_view;
    if (r_view < 0)
        r_view = r_view + num_views;
    int g_view = r_view + 1;
    int b_view = r_view + 2;
    if (g_view >= num_views)
        g_view = g_view - num_views;
    if (b_view >= num_views)
        b_view =  b_view - num_views;
    
    // Write to Output
    int b_out = (tx + ty * num_cols_out) * elem_sz;
    int g_out = b_out + 1;
    int r_out = g_out + 1;

    output[b_out] = fast_bilinear_interp(views[b_view], elem_sz, 0, x_samp, y_samp, num_cols_in, num_rows_in);
    output[g_out] = fast_bilinear_interp(views[g_view], elem_sz, 1, x_samp, y_samp, num_cols_in, num_rows_in);
    output[r_out] = fast_bilinear_interp(views[r_view], elem_sz, 2, x_samp, y_samp, num_cols_in, num_rows_in);
}

void d_mux_multiview( unsigned char **d_views, unsigned char* d_out_data, 
                      int num_views, float angle, 
				      int num_rows_in, int num_cols_in, int num_rows_out, int num_cols_out, int elem_sz)
{
	// Setup Block & Grid Size
    size_t bw = 32;
    size_t bh = 32;
    size_t gw = (num_cols_out + bw - 1) / bw;
    size_t gh = (num_rows_out + bh - 1) / bh;
    
    int kernel_num = 0;
    if (num_rows_out % num_views == 0)
    {
        bw = num_cols_out / num_views;
        bh = 1;
        gw = (num_cols_out + bw - 1) / bw / num_views;
        gh = (num_rows_out + bh - 1) / bh;
        kernel_num = 1;
    }
    
    const dim3 block_sz(bw, bh, 1);
    const dim3 grid_sz(gw, gh, 1);
    
    float y_interval = (float) num_views / tan(angle * PI / 180.0) / (float) elem_sz;
    
    if (kernel_num == 0)
        mux_multiview_kernel<<<grid_sz, block_sz>>>(d_views, d_out_data, num_views, y_interval, num_rows_in, num_cols_in, num_rows_out, num_cols_out, elem_sz);
    else if (kernel_num == 1) 
        mux_multiview_kernel_2<<<grid_sz, block_sz>>>(d_views, d_out_data, num_views, y_interval, 1.0f/y_interval, num_rows_in, num_cols_in, num_rows_out, num_cols_out, elem_sz);
    cudaDeviceSynchronize();
}


void mux_multiview( unsigned char **views, unsigned char* out_data, 
                      int num_views, float angle, 
				      int num_rows_in, int num_cols_in, int num_rows_out, int num_cols_out, int elem_sz)
{
    cudaEventPair_t timer;
	// Memory Allocation of Input
	unsigned char** d_views;
	checkCudaError(cudaMalloc(&d_views, sizeof(unsigned char *) * num_views));
	
	unsigned char** h_views = (unsigned char**) malloc(sizeof(unsigned char*) * num_views);
	for (int v = 0; v < num_views; ++v)
	{
		checkCudaError(cudaMalloc(&h_views[v], sizeof(unsigned char) * num_cols_in * num_rows_in * elem_sz));
		checkCudaError(cudaMemcpy(h_views[v], views[v], sizeof(unsigned char) * num_cols_in * num_rows_in * elem_sz, cudaMemcpyHostToDevice));
	}
	checkCudaError(cudaMemcpy(d_views, h_views, sizeof(unsigned char *) * num_views, cudaMemcpyHostToDevice));

	// Memory Allocation of Output
	unsigned char* d_out_data;
	checkCudaError(cudaMalloc(&d_out_data, sizeof(unsigned char) * num_cols_out * num_rows_out * elem_sz));
    
	// Setup Block & Grid Size
    size_t bw = 32;
    size_t bh = 32;
    size_t gw = (num_cols_out + bw - 1) / bw;
    size_t gh = (num_rows_out + bh - 1) / bh;
    
    int kernel_num = 0;
    if (num_rows_out % num_views == 0)
    {
        bw = num_cols_out / num_views;
        bh = 1;
        gw = (num_cols_out + bw - 1) / bw / num_views;
        gh = (num_rows_out + bh - 1) / bh;
        kernel_num = 1;
    }
    
    const dim3 block_sz(bw, bh, 1);
    const dim3 grid_sz(gw, gh, 1);
    
    float y_interval = (float) num_views / tan(angle * PI / 180.0) / (float) elem_sz;

    if (kernel_num == 0)
    {
        startCudaTimer(&timer);
        mux_multiview_kernel<<<grid_sz, block_sz>>>(d_views, d_out_data, num_views, y_interval, num_rows_in, num_cols_in, num_rows_out, num_cols_out, elem_sz);
        stopCudaTimer(&timer, "Multiview Interlace Kernel");
    }
    else if (kernel_num == 1) 
    {
        startCudaTimer(&timer);
        mux_multiview_kernel_2<<<grid_sz, block_sz>>>(d_views, d_out_data, num_views, y_interval, 1.0f/y_interval, num_rows_in, num_cols_in, num_rows_out, num_cols_out, elem_sz);
        stopCudaTimer(&timer, "Multiview Interlace Kernel #2");
    }

	// Copy Memory back to Host
	checkCudaError(cudaMemcpy(out_data, d_out_data, sizeof(unsigned char) * num_cols_out * num_rows_out * elem_sz,cudaMemcpyDeviceToHost));

	// De-allocation of Host & Device Memory
	for (int v = 0; v < num_views; ++v)
		cudaFree(h_views[v]);
	cudaFree(d_views);
	cudaFree(d_out_data);
    free(h_views);
}

#endif
