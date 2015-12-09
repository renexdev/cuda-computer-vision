NVCC = /usr/local/cuda/bin/nvcc
C_COMPILER_FLAGS = `pkg-config --cflags --libs opencv` -lopencv_gpu -lopencv_core -lopencv_objdetect -lopencv_imgproc -lopencv_highgui

CudaTestKevin: CudaTestKevin.cu
	$(NVCC) $^ -o $@ $(C_COMPILER_FLAGS)

clean:
	rm CudaTestKevin
