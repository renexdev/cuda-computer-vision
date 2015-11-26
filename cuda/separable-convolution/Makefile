NVCC = /usr/local/cuda/bin/nvcc
NVCC_FLAGS = -g -G -Xcompiler -Wall `pkg-config --cflags --libs opencv`

separable_convolution: separable_convolution.cu
	$(NVCC) $(NVCC_FLAGS) $^ -o $@
