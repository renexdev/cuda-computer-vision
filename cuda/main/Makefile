NVCC = /usr/local/cuda/bin/nvcc
NVCC_FLAGS = -g -G -Xcompiler -Wall
C_COMPILER_FLAGS = `pkg-config --cflags --libs opencv` 
HEADER_FILES = ../separable-convolution/separable_convolution.cu ../edge-detect/edge_detect.cu ../motion-tracking/motion_tracking.cu

main: main.cu
	$(NVCC) $(NVCC_FLAGS) $^ -o $@ $(HEADER_FILES) $(C_COMPILER_FLAGS)

clean:
	rm main
