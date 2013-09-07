#ifndef D_DIBR_OCCL_H
#define D_DIBR_OCCL_H
#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include "d_mux_common.h"

__global__ void dibr_occl_to_mask_kernel(float *mask, unsigned char *occl,
                                         int num_rows, int num_cols);

void d_dibr_occl_to_mask(float *d_mask_l, float *d_mask_r,
                         unsigned char* d_occl_l, unsigned char* d_occl_r,
                         int num_rows, int num_cols);

void dibr_occl_to_mask(float *mask_l, float *mask_r,
                       unsigned char* occl_l, unsigned char* occl_r,
                       int num_rows, int num_cols);

__global__ void dibr_find_occlusion_kernel(unsigned char *occl, float *disp,
                                           int dir,
                                           int num_rows, int num_cols);


void d_dibr_occl(unsigned char* d_occl_l, unsigned char* d_occl_r,
                 float* d_disp_l, float* d_disp_r,
                 int num_rows, int num_cols);

void dibr_occl(unsigned char* occl_l, unsigned char* occl_r,
               float* disp_l, float* disp_r,
               int num_rows, int num_cols);

#endif
