NVCC = /usr/local/cuda/bin/nvcc
NVCC_FLAGS = -g -G -Xcompiler -Wall

motion_tracking: motion_tracking.cu
	$(NVCC) $(NVCC_FLAGS) $^ -o $@
