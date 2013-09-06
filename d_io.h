#ifndef D_IO_H
#define D_IO_H
// User Sources
#include "d_dibr_fwarp.h"
#include "d_dibr_bwarp.h"
#include "d_dibr_occl.h"
#include "d_dc_wta.h"
#include "d_dc_hslo.h"
#include "d_ca_cross.h"
#include "d_ci_adcensus.h"
#include "d_ci_census.h"
#include "d_ci_ad.h"
#include "d_tx_scale.h"
#include "d_mux_multiview.h"
#include "d_demux_common.h"
#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

void adcensus_stm(unsigned char *img_sbs, float *disp_l, float *disp_r,
                  unsigned char** views, unsigned char* interlaced,
                  int num_rows, int num_cols_sbs, int num_cols, 
                  int num_rows_out, int num_cols_out, int elem_sz,
                  int num_views, int angle,
                  int num_disp, int zero_disp,
                  float ad_coeff, float census_coeff,
                  float ucd, float lcd, int usd, int lsd);

#endif
