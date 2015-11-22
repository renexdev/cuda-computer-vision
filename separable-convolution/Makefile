NVCC = /usr/local/cuda/bin/nvcc
NVCC_FLAGS = -g -G -Xcompiler -Wall

separable_convolution: separable_convolution.cu
	$(NVCC) $(NVCC_FLAGS) $^ -o $@
