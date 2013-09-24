# CUDA PATHS
CUDA_PATH ?= /usr/local/cuda
CUDA_LIB_PATH = $(CUDA_PATH)/lib
CUDA_INCLUDE_PATH = $(CUDA_PATH)/include

# OPENCV PATHS
OPENCV_LIB_PATH=/usr/local/lib
OPENCV_INCLUDE_PATH=/usr/local/include

# OPENCV LIBS
OPENCV_LIBS=-lopencv_core -lopencv_imgproc -lopencv_highgui
USER_LIBS= -lm

NVCC_OPTS = -O3 -arch=sm_30 -Xcompiler -Wall -Xcompiler -Wextra -m64
GCC_OPTS =  -O3 -Wall -Wextra -m64

NVCC=nvcc
GCC=g++

DEVICE_OBJECTS = d_io.o d_alu.o d_ci_census.o d_ci_ad.o d_mux_multiview.o d_tx_scale.o d_ci_adcensus.o d_ca_cross_sum.o d_ca_cross.o d_dc_wta.o d_dibr_fwarp.o d_dibr_bwarp.o d_dibr_occl.o d_mux_common.o d_dc_hslo.o d_demux_common.o d_filter.o d_filter_bilateral.o d_filter_gaussian.o d_op.o d_dr_dcc.o d_dr_irv.o

DEVICE_LINK = device.o
HOST_OBJECTS = getCPUtime.o

all: video_io image_io

# Link Host to Device Objects
video_io: host_video_io.o $(HOST_OBJECTS) $(DEVICE_LINK)
	$(GCC) $(GCC_OPTS) -o video_io host_video_io.o $(HOST_OBJECTS) $(DEVICE_OBJECTS) $(DEVICE_LINK) -L$(OPENCV_LIB_PATH) -L$(CUDA_LIB_PATH) $(OPENCV_LIBS) $(USER_LIBS) -lcudart

image_io: host_image_io.o $(HOST_OBJECTS) $(DEVICE_LINK)
	$(GCC) $(GCC_OPTS) -o image_io host_image_io.o $(HOST_OBJECTS) $(DEVICE_OBJECTS) $(DEVICE_LINK) -L$(OPENCV_LIB_PATH) -L$(CUDA_LIB_PATH) $(OPENCV_LIBS) $(USER_LIBS) -lcudart

# Host Objects
host_video_io.o: video_io.cpp
	$(GCC) -c video_io.cpp $(GCC_OPTS) -I $(CUDA_INCLUDE_PATH) -I $(OPENCV_INCLUDE_PATH) -o host_video_io.o

host_image_io.o: image_io.cpp
	$(GCC) -c image_io.cpp $(GCC_OPTS) -I $(CUDA_INCLUDE_PATH) -I $(OPENCV_INCLUDE_PATH) -o host_image_io.o

getCPUtime.o: getCPUtime.cpp
	$(GCC) -c getCPUtime.cpp $(GCC_OPTS) -I $(CUDA_INCLUDE_PATH) -I $(OPENCV_INCLUDE_PATH) -o getCPUtime.o

# Device Link
device.o: $(DEVICE_OBJECTS)
	$(NVCC) $(NVCC_OPTS) -dlink $(DEVICE_OBJECTS) -o device.o

# Device Objects
d_dr_irv.o: d_dr_irv.cu d_dr_irv.h cuda_utils.h
	$(NVCC) -dc d_dr_irv.cu $(NVCC_OPTS) -I $(CUDA_INCLUDE_PATH) -I $(OPENCV_INCLUDE_PATH)

d_dr_dcc.o: d_dr_dcc.cu d_dr_dcc.h cuda_utils.h
	$(NVCC) -dc d_dr_dcc.cu $(NVCC_OPTS) -I $(CUDA_INCLUDE_PATH) -I $(OPENCV_INCLUDE_PATH)

d_op.o: d_op.cu d_op.h cuda_utils.h
	$(NVCC) -dc d_op.cu $(NVCC_OPTS) -I $(CUDA_INCLUDE_PATH) -I $(OPENCV_INCLUDE_PATH)

d_filter_bilateral.o: d_filter_bilateral.cu d_filter_bilateral.h cuda_utils.h
	$(NVCC) -dc d_filter_bilateral.cu $(NVCC_OPTS) -I $(CUDA_INCLUDE_PATH) -I $(OPENCV_INCLUDE_PATH)

d_filter_gaussian.o: d_filter_gaussian.cu d_filter_gaussian.h cuda_utils.h
	$(NVCC) -dc d_filter_gaussian.cu $(NVCC_OPTS) -I $(CUDA_INCLUDE_PATH) -I $(OPENCV_INCLUDE_PATH)

d_filter.o: d_filter.cu d_filter.h cuda_utils.h
	$(NVCC) -dc d_filter.cu $(NVCC_OPTS) -I $(CUDA_INCLUDE_PATH) -I $(OPENCV_INCLUDE_PATH)

d_io.o: d_io.cu d_io.h cuda_utils.h
	$(NVCC) -dc d_io.cu $(NVCC_OPTS) -I $(CUDA_INCLUDE_PATH) -I $(OPENCV_INCLUDE_PATH)

d_dibr_occl.o: d_dibr_occl.cu d_dibr_occl.h cuda_utils.h
	$(NVCC) -dc d_dibr_occl.cu $(NVCC_OPTS) -I $(CUDA_INCLUDE_PATH) -I $(OPENCV_INCLUDE_PATH)

d_dibr_fwarp.o: d_dibr_fwarp.cu d_dibr_fwarp.h cuda_utils.h
	$(NVCC) -dc d_dibr_fwarp.cu $(NVCC_OPTS) -I $(CUDA_INCLUDE_PATH) -I $(OPENCV_INCLUDE_PATH)

d_dibr_bwarp.o: d_dibr_bwarp.cu d_dibr_bwarp.h cuda_utils.h
	$(NVCC) -dc d_dibr_bwarp.cu $(NVCC_OPTS) -I $(CUDA_INCLUDE_PATH) -I $(OPENCV_INCLUDE_PATH)

d_dc_hslo.o: d_dc_hslo.cu d_dc_hslo.h cuda_utils.h
	$(NVCC) -dc d_dc_hslo.cu $(NVCC_OPTS) -I $(CUDA_INCLUDE_PATH) -I $(OPENCV_INCLUDE_PATH)

d_dc_wta.o: d_dc_wta.cu d_dc_wta.h cuda_utils.h
	$(NVCC) -dc d_dc_wta.cu $(NVCC_OPTS) -I $(CUDA_INCLUDE_PATH) -I $(OPENCV_INCLUDE_PATH)

d_ca_cross_sum.o: d_ca_cross_sum.cu d_ca_cross_sum.h cuda_utils.h
	$(NVCC) -dc d_ca_cross_sum.cu $(NVCC_OPTS) -I $(CUDA_INCLUDE_PATH) -I $(OPENCV_INCLUDE_PATH)

d_ca_cross.o: d_ca_cross.cu d_ca_cross.h cuda_utils.h
	$(NVCC) -dc d_ca_cross.cu $(NVCC_OPTS) -I $(CUDA_INCLUDE_PATH) -I $(OPENCV_INCLUDE_PATH)

d_ci_adcensus.o: d_ci_adcensus.cu d_ci_adcensus.h cuda_utils.h d_ci_ad.o
	$(NVCC) -dc d_ci_adcensus.cu $(NVCC_OPTS) -I $(CUDA_INCLUDE_PATH) -I $(OPENCV_INCLUDE_PATH)

d_ci_census.o: d_ci_census.cu d_ci_census.h cuda_utils.h 
	$(NVCC) -dc d_ci_census.cu $(NVCC_OPTS) -I $(CUDA_INCLUDE_PATH) -I $(OPENCV_INCLUDE_PATH)

d_ci_ad.o: d_ci_ad.cu d_ci_ad.h cuda_utils.h 
	$(NVCC) -dc d_ci_ad.cu $(NVCC_OPTS) -I $(CUDA_INCLUDE_PATH) -I $(OPENCV_INCLUDE_PATH)

d_mux_multiview.o: d_mux_multiview.cu d_mux_multiview.h cuda_utils.h d_alu.o 
	$(NVCC) -dc d_mux_multiview.cu $(NVCC_OPTS) -I $(CUDA_INCLUDE_PATH) -I $(OPENCV_INCLUDE_PATH)

d_mux_common.o: d_mux_common.cu d_mux_common.h cuda_utils.h
	$(NVCC) -dc d_mux_common.cu $(NVCC_OPTS) -I $(CUDA_INCLUDE_PATH) -I $(OPENCV_INCLUDE_PATH)

d_demux_common.o: d_demux_common.cu d_demux_common.h cuda_utils.h
	$(NVCC) -dc d_demux_common.cu $(NVCC_OPTS) -I $(CUDA_INCLUDE_PATH) -I $(OPENCV_INCLUDE_PATH)

d_tx_scale.o: d_tx_scale.cu d_tx_scale.h cuda_utils.h d_alu.o
	$(NVCC) -dc d_tx_scale.cu $(NVCC_OPTS) -I $(CUDA_INCLUDE_PATH) -I $(OPENCV_INCLUDE_PATH)

d_alu.o: d_alu.cu d_alu.h cuda_utils.h
	$(NVCC) -dc d_alu.cu $(NVCC_OPTS) -I $(CUDA_INCLUDE_PATH) -I $(OPENCV_INCLUDE_PATH)

clean:
	rm -f *.o video_io image_io *.png
