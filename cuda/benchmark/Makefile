NVCC = /usr/local/cuda/bin/nvcc
NVCC_FLAGS = -g -G -Xcompiler -Wall `pkg-config --cflags --libs opencv`

benchmark: benchmark.cu
	$(NVCC) $(NVCC_FLAGS) $^ -o $@
