#ifndef D_IO_H
#define D_IO_H
#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <math.h>

#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>

#include "cuda_utils.h"

#include "d_dr_dcc.h"
#include "d_dr_irv.h"
#include "d_filter_gaussian.h"
#include "d_filter_bilateral.h"
#include "d_filter.h"
#include "d_dibr_occl.h"
#include "d_dibr_fwarp.h"
#include "d_dibr_bwarp.h"
#include "d_dc_wta.h"
#include "d_dc_hslo.h"
#include "d_ca_cross.h"
#include "d_ci_adcensus.h"
#include "d_ci_census.h"
#include "d_ci_ad.h"
#include "d_tx_scale.h"
#include "d_mux_multiview.h"
#include "d_demux_common.h"

void adcensus_stm(unsigned char *img_sbs, float *disp_l, float *disp_r,
                  unsigned char* interlaced,
                  int num_rows, int num_cols_sbs, int num_cols, 
                  int num_rows_out, int num_cols_out, int elem_sz,
                  int num_views, int angle,
                  int num_disp, int zero_disp,
                  float ad_coeff, float census_coeff,
                  float ucd, float lcd, int usd, int lsd,
                  int thresh_s, float thresh_h);

void adcensus_stm_2(unsigned char *img_sbs, float *disp_l, float *disp_r,
                    unsigned char* interlaced,
                    int num_rows, int num_cols_sbs, int num_cols, 
                    int num_rows_out, int num_cols_out, 
					int num_rows_disp, int num_cols_disp,	
					int elem_sz, float disp_scale,
                    int num_views, int angle,
                    int num_disp, int zero_disp,
                    float ad_coeff, float census_coeff,
                    float ucd, float lcd, int usd, int lsd,
                    int thresh_s, float thresh_h);

#endif
