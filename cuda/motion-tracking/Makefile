NVCC = /usr/local/cuda/bin/nvcc
NVCC_FLAGS = -g -G -Xcompiler -Wall `pkg-config --cflags --libs opencv`
HEADER_FILES = ../separable-convolution/separable_convolution.cu ../edge-detect/edge_detect.cu

motion_tracking: motion_tracking.cu
	$(NVCC) $(NVCC_FLAGS) $^ -o $@ $(HEADER_FILES)

clean:
	rm motion_tracking
