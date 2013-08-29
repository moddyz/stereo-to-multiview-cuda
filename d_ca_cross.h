#ifndef D_CA_CROSS_H
#define D_CA_CROSS_H
#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

void ca_cross(unsigned char* img, float** cost, float** acost,
              float ucd, float lcd, int usd, int lsd,
              int num_disp, int num_rows, int num_cols, int elem_sz);

#endif
