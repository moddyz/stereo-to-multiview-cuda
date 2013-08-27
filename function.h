#ifndef functkernel_h
#define functkernel_h
#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

void histogram_process(float *h_img, float &h_min, float &h_max,  unsigned int *h_cdf, size_t num_cols, size_t num_rows, size_t num_bins);

#endif

