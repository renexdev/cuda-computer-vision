#include <stdio.h>
#include <sys/time.h>
#include <cuda_runtime.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include "separable_convolution.h"
#include "../helper/helper_cuda.h"

using namespace cv;

#define TX 16
#define TY 16


__global__ void horizontal_convolve(int *d_out, int *x, int *h, int x_width, int x_height, int h_width, int h_height) {
    const int r = blockIdx.y * blockDim.y + threadIdx.y;
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    const int i = r * (x_width + h_width - 1) + c;
    
    int sum = 0;
    for (int j = 0; j < h_width; j++) {
        int p = x_width*r + c - j;
        if (c - j >= 0 && c - j < x_width) {
            sum += h[j] * x[p];
        }
    }
    d_out[i] = sum;
    __syncthreads();
}

__global__ void vertical_convolve(int *d_out, int *x, int *h, int x_width, int x_height, int h_width, int h_height, double constant_scalar) {
	const int r = blockIdx.y * blockDim.y + threadIdx.y;
	const int c = blockIdx.x * blockDim.x + threadIdx.x;
	const int i = r * x_width + c;

    int sum = 0;
    for (int j = 0; j < h_height; j++) {
        int p = x_width*(r - j) + c;
        if (r - j >= 0 && r - j < x_height) {
            sum += h[j] * x[p];
        }
    }
    d_out[i] = (int)(constant_scalar * (double)sum);
    __syncthreads();
}

void serial_separable_convolve(int *out, int *x, int *horizontal_filter, int *vertical_filter, int x_width, int x_height, int horizontal_filter_width, int vertical_filter_height, double constant_scalar) {
    int *horizontal_out = (int *)malloc((x_width + horizontal_filter_width - 1) * x_height * sizeof(int));
    for (int m = 0; m < x_height; m++) {
        for (int n = 0; n < x_width + horizontal_filter_width - 1; n++) {
            int h_sum = 0;
            for (int j = 0; j < horizontal_filter_width; j++) {
                if (n - j >= 0 && n - j < x_width) {
                    h_sum += x[m * x_width + n - j] * horizontal_filter[j];
                }
            }
            horizontal_out[m * (x_width + horizontal_filter_width - 1) + n] = h_sum;
        }
    }
    for (int v_m = 0; v_m < x_height + vertical_filter_height - 1; v_m++) {
        for (int v_n = 0; v_n < x_width + horizontal_filter_width - 1; v_n++) {
            int v_sum = 0;
            for (int i = 0; i < vertical_filter_height; i++) {
                if (v_m - i >= 0 && v_m - i < x_height) {
                    v_sum += horizontal_out[(v_m - i) * (x_width + horizontal_filter_width - 1) + v_n] * vertical_filter[i];
                }
            }
            out[v_m * (x_width + horizontal_filter_width - 1) + v_n] = (int)(constant_scalar * (double)v_sum);
        }
    }
}

void serial_naive_convolve(int *out, int *x, int *h, int x_width, int x_height, int h_width, int h_height) {
    for (int m = 0; m < x_height + h_height - 1; m++) {
        for (int n = 0; n < x_width + h_width - 1; n++) {
            int sum = 0;
            for (int i = 0; i < h_height; i++) {
                for (int j = 0; j < h_width; j++) {
                    if (m - i >= 0 && m - i < x_height && n - j >= 0 && n - j < x_width) {
                        sum += h[i * h_width + j] * x[(m - i) * x_width + n - j];
                    }
                }
            }
            out[m * (x_width + h_width - 1) + n] = sum;
        }
    }
}

void separable_convolve(int *output, int *x, int x_width, int x_height, int *horizontal_filter, int *vertical_filter, int kernel_size, double constant_scalar) {
    // Specify lengths of filters and input
    int horizontal_filter_width = kernel_size;
    int vertical_filter_height = kernel_size;

    // Allocate space for host and device arrays
    static int *dev_horizontal_out, *dev_vertical_out;  // Results of the horizontal and vertical convolutions on the input array
    static int *dev_horizontal_filter, *dev_vertical_filter, *dev_x;  // Horizontal filter, vertical filter, and input array

    // Horizontal filter, followed by vertical filter
    int horizontal_convolution_width = x_width + horizontal_filter_width - 1;
    int horizontal_convolution_height = x_height;
    int vertical_convolution_width = horizontal_convolution_width;
    int vertical_convolution_height = horizontal_convolution_height + vertical_filter_height - 1;

    // Allocate space for horizontal result, vertical result, horizontal filter, vertical filter, and input
    checkCudaErrors(cudaMalloc(&dev_horizontal_out, horizontal_convolution_width*horizontal_convolution_height*sizeof(int)));
    checkCudaErrors(cudaMalloc(&dev_vertical_out, vertical_convolution_width*vertical_convolution_height*sizeof(int)));
    checkCudaErrors(cudaMalloc(&dev_horizontal_filter, horizontal_filter_width*sizeof(int)));
    checkCudaErrors(cudaMalloc(&dev_vertical_filter, vertical_filter_height*sizeof(int)));
    checkCudaErrors(cudaMalloc(&dev_x, x_width*x_height*sizeof(int)));

    // Copy host arrays to device
    checkCudaErrors(cudaMemcpy(dev_horizontal_filter, horizontal_filter, horizontal_filter_width*sizeof(int), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(dev_vertical_filter, vertical_filter, vertical_filter_height*sizeof(int), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(dev_x, x, x_width*x_height*sizeof(int), cudaMemcpyHostToDevice));

    // Initialize grid
    dim3 block_size(TX, TY);
	int bx_horizontal = horizontal_convolution_width/block_size.x;
	int by_horizontal = horizontal_convolution_height/block_size.y;
	dim3 grid_size_horizontal = dim3(bx_horizontal, by_horizontal);
	int bx_vertical = vertical_convolution_width/block_size.x;
	int by_vertical = vertical_convolution_height/block_size.y;
	dim3 grid_size_vertical = dim3(bx_vertical, by_vertical);
    
    horizontal_convolve<<<grid_size_horizontal, block_size>>>(dev_horizontal_out, dev_x, dev_horizontal_filter, x_width, x_height, horizontal_filter_width, 1);
    vertical_convolve<<<grid_size_vertical, block_size>>>(dev_vertical_out, dev_horizontal_out, dev_vertical_filter, horizontal_convolution_width, horizontal_convolution_height, 1, vertical_filter_height, constant_scalar);

    // Copy result data from device to host
    checkCudaErrors(cudaMemcpy(output, dev_vertical_out, vertical_convolution_width * vertical_convolution_height * sizeof(int), cudaMemcpyDeviceToHost));

    // Responsible programmer
    checkCudaErrors(cudaFree(dev_vertical_out));
    checkCudaErrors(cudaFree(dev_horizontal_out));
    checkCudaErrors(cudaFree(dev_horizontal_filter));
    checkCudaErrors(cudaFree(dev_vertical_filter));
    checkCudaErrors(cudaFree(dev_x));
}
