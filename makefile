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

HOST_OBJECTS = host.o 
DEVICE_OBJECTS = d_alu.o d_ci_census.o d_ci_ad.o d_mux_multiview.o d_tx_scale.o d_ci_adcensus.o d_ca_cross.o d_dc_wta.o d_dibr_warp.o d_mux_common.o d_dc_hslo.o d_demux_common.o
DEVICE_LINK = device.o

# Link Host to Device Objects
program: $(HOST_OBJECTS) $(DEVICE_LINK)
	$(GCC) $(GCC_OPTS) -o program $(HOST_OBJECTS) $(DEVICE_OBJECTS) $(DEVICE_LINK) -L$(OPENCV_LIB_PATH) -L$(CUDA_LIB_PATH) $(OPENCV_LIBS) $(USER_LIBS) -lcudart

# Host Objects
host.o: main.cpp
	$(GCC) -c main.cpp $(GCC_OPTS) -I $(CUDA_INCLUDE_PATH) -I $(OPENCV_INCLUDE_PATH) -o host.o

# Device Link
device.o: $(DEVICE_OBJECTS)
	$(NVCC) $(NVCC_OPTS) -dlink $(DEVICE_OBJECTS) -o device.o

# Device Objects
d_dibr_warp.o: d_dibr_warp.cu d_dibr_warp.h cuda_utils.h
	$(NVCC) -dc d_dibr_warp.cu $(NVCC_OPTS) -I $(CUDA_INCLUDE_PATH) -I $(OPENCV_INCLUDE_PATH)

d_dc_hslo.o: d_dc_hslo.cu d_dc_hslo.h cuda_utils.h
	$(NVCC) -dc d_dc_hslo.cu $(NVCC_OPTS) -I $(CUDA_INCLUDE_PATH) -I $(OPENCV_INCLUDE_PATH)

d_dc_wta.o: d_dc_wta.cu d_dc_wta.h cuda_utils.h
	$(NVCC) -dc d_dc_wta.cu $(NVCC_OPTS) -I $(CUDA_INCLUDE_PATH) -I $(OPENCV_INCLUDE_PATH)

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
	rm -f *.o program *.png
