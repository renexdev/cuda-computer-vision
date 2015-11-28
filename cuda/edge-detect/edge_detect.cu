#include <stdio.h>
#include <cuda.h>
#include <sys/time.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include "../separable-convolution/separable_convolution.h"

using namespace cv;


__global__ void thresholding_maximum_suppression(int *dev_output, int input_width, int input_height, int *g_x, int *g_y, int high_threshold, int low_threshold) {
    const int r = blockIdx.x;
    const int c = threadIdx.x;
    const int i = r * blockDim.x + c;

    dev_output[i] = 255;
    printf("%d, %d\n", r, c);

    if (0 && r > 1 && c > 1 && r < input_height - 1 && c < input_width - 1) {
        double magnitude = __dsqrt_rd(g_x[i] * g_x[i] + g_y[i] * g_y[i]);
        double magnitude_above = __dsqrt_rd(g_x[i] * g_x[i] + g_y[(r - 1) * input_width + c] * g_y[(r - 1) * input_width + c]);
        double magnitude_below = __dsqrt_rd(g_x[i] * g_x[i] + g_y[(r + 1) * input_width + c] * g_y[(r + 1) * input_width + c]);
        double magnitude_left = __dsqrt_rd(g_x[r * input_width + c - 1] * g_x[r * input_width + c - 1] + g_y[i] * g_y[i]);
        double magnitude_right = __dsqrt_rd(g_x[r * input_width + c + 1] * g_x[r * input_width + c + 1] + g_y[i] * g_y[i]);
        double theta = atan2((double)g_y[i], (double)g_x[i]);
        printf("r %d, c %d, mag %d, theta %d\n", r, c, magnitude, theta);
    
        int vertical_check = (M_PI/3.0 < theta && theta > 2.0*M_PI/3.0) || (-2.0*M_PI/3.0 < theta && theta < -M_PI/3);
        int is_vertical_max = vertical_check && magnitude > magnitude_below && magnitude > magnitude_above;
        int horizontal_check = (-M_PI/6.0 < theta && theta < M_PI/6.0) || (-M_PI < theta && theta < -5.0*M_PI/6.0) || (5*M_PI/6.0 < theta && theta < M_PI);
        int is_horizontal_max = horizontal_check && magnitude > magnitude_right && magnitude > magnitude_left;
        int diagonal_check = !vertical_check && !horizontal_check;
        // TODO: finish diagonal
    
        // TODO: move high threshold check outside to eliminate unnecessary direction check computations
        if ((is_vertical_max || is_horizontal_max) && magnitude > high_threshold) {
            for (int m = -2; m < 2; m++) {
                for (int n = -2; n < 2; n++) {
                    if (__dsqrt_rd(g_x[(r + m) * input_width + c + n] * g_x[(r + m) * input_width + c + n] + g_y[(r + m) * input_width + c + n] * g_y[(r + m) * input_width + c + n])) {
                        if (r + m > 0 && r + m < input_height && c + n > 0 && c + n < input_width) {
                            dev_output[(r + m) * input_width + c + n] = 255;
                        }
                    }
                }
            }
        }
    }
    __syncthreads();
}

int main() {
    // Load external image into array
    Mat image = imread("../images/keyboard_1000_1000.jpg", 0);
    int *x = (int *)malloc(image.cols * image.rows * sizeof(int));
    static int *gaussian_out;
    gaussian_out = (int *)malloc((image.rows * image.cols + 10) * sizeof(int));  // TODO: make this more precise
    for (int i = 0; i < image.rows; i++) {
        for (int j = 0; j < image.cols; j++) {
            x[i * image.rows + j] = image.at<uchar>(i, j);
        }
    }

    // Gaussian filter
    int horizontal_filter[3] = {1, 2, 1};
    int vertical_filter[3] = {1, 2, 1};
    int kernel_size = 3;
    double constant_scalar = 1.0/16.0;
    separable_convolve(gaussian_out, x, image.cols, image.rows, horizontal_filter, vertical_filter, kernel_size, constant_scalar);
    int gaussian_out_width = image.cols + kernel_size - 1;
    int gaussian_out_height = image.rows + kernel_size - 1;

    // Sobel filter
    int sobel_out_width = gaussian_out_width + kernel_size - 1;
    int sobel_out_height = gaussian_out_height + kernel_size - 1;
    // Horizontal direction
    int gx_horizontal[3] = {1, 0, -1};
    int gx_vertical[3] = {1, 2, 1};
    static int *gx_out = (int *)malloc(sobel_out_width * sobel_out_height * sizeof(int));
    separable_convolve(gx_out, gaussian_out, sobel_out_width, sobel_out_height, gx_horizontal, gx_vertical, 3, 1);
    // Vertical direction
    int gy_horizontal[3] = {1, 2, 1};
    int gy_vertical[3] = {1, 0, -1};
    static int *gy_out = (int *)malloc(sobel_out_width * sobel_out_height * sizeof(int));
    separable_convolve(gy_out, gaussian_out, sobel_out_width, sobel_out_height, gy_horizontal, gy_vertical, 3, 1);

    // Magnitude and thresholding
    static int *edges = (int *)malloc(sobel_out_width * sobel_out_height * sizeof(int));
    int *dev_edges, *dev_gx, *dev_gy;
    cudaMalloc(&dev_edges, sobel_out_width * sobel_out_height * sizeof(int));
    cudaMalloc(&dev_gx, sobel_out_width * sobel_out_height * sizeof(int));
    cudaMalloc(&dev_gy, sobel_out_width * sobel_out_height * sizeof(int));
    cudaMemcpy(dev_gx, gx_out, sobel_out_width * sobel_out_height * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_gy, gy_out, sobel_out_width * sobel_out_height * sizeof(int), cudaMemcpyHostToDevice);
    thresholding_maximum_suppression<<<sobel_out_height, sobel_out_width>>>(edges, sobel_out_width, sobel_out_height, dev_gx, dev_gy, 120, 100);
    cudaMemcpy(edges, dev_edges, sobel_out_width * sobel_out_height * sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(dev_edges);
    cudaFree(dev_gx);
    cudaFree(dev_gy);

    // Write to disk
    Mat edges_image(sobel_out_width, sobel_out_height, CV_32SC1, edges);
    imwrite("temp.jpg", edges_image);

    return 0;
}
