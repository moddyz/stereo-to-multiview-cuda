#ifndef D_DC_WTA_H
#define D_DC_WTA_H
#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

void dc_wta(float** cost, float* disp, int num_disp, int zero_disp, int num_rows, int num_cols);


#endif
