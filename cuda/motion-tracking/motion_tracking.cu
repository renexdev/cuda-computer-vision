#include <stdio.h>
#include <sys/time.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>

using namespace cv;


__global__ void difference_filter(int *dev_out, int *edges_1, int *edges_2, int width, int height, int threshold) {
    // Note: width should correspond to width of dev_out, edges_1, and edges_2; same for height
    const int r = blockIdx.x;
    const int c = threadIdx.x;
    const int i = r * blockDim.x + c;

    // Set it to 0 initially
    dev_out[i] = 0;
    if (edges_1[i] != edges_2[i]) {
        // Set to 1 if there is a pixel mismatch
        dev_out[i] = 1;
        for (int x_apron = -threshold; x_apron <= threshold; x_apron++) {
            for (int y_apron = -threshold; y_apron <= threshold; y_apron++) {
                // Ensure the requested index is within bounds of image
                if (c + x_apron > 0 && r + y_apron > 0 && c + x_apron < width && r + y_apron < height) {
                    // Check if there is a matching pixel in the apron, within the threshold
                    if (edges_1[(r + y_apron) * width + c + x_apron] == edges_2[i]) {
                        // Set it back to 0 if a corresponding pixel exists within the vicinity of the match
                        dev_out[i] = 0;
                    }
                }
            }
        }
    }
}

double serial_difference_filter(int *difference, int *edges_1, int *edges_2, int width, int height, int threshold) {
    struct timeval tv1, tv2;
    gettimeofday(&tv1, NULL);
    for (int x = 0; x < width; x++) {
        for (int y = 0; y < height; y++) {
            difference[y * width + x] = 0;
            if (edges_1[y * width + x] != edges_2[y * width + x]) {
                difference[y * width + x] = 1;
                for (int x_apron = -threshold; x_apron <= threshold; x_apron++) {
                    for (int y_apron = -threshold; y_apron <= threshold; y_apron++) {
                        if (x + x_apron > 0 && y + y_apron > 0 && x + x_apron < width && y + y_apron < height) {
                            if (edges_1[(y + y_apron) * width + x + x_apron] == edges_2[y * width + x]) {
                                difference[y * width + x] = 0;
                            }
                        }
                    }
                }
            }
        }
    }
    gettimeofday(&tv2, NULL);
    double time_spent = (double)(tv2.tv_usec - tv1.tv_usec) / 1000000 + (double)(tv2.tv_sec - tv1.tv_sec);
    printf("Serial difference filter execution time: %f seconds\n", time_spent);
    return time_spent;
}

void motion_track(int *output, int *edges_1, int *edges_2, int width, int height, int threshold) {
    // Allocate space on device
    int *dev_edges_1, *dev_edges_2, *dev_out;
    cudaMalloc(&dev_out, width*height*sizeof(int));
    cudaMalloc(&dev_edges_1, width*height*sizeof(int));
    cudaMalloc(&dev_edges_2, width*height*sizeof(int));

    // Copy host arrays to device
    cudaMemcpy(dev_edges_1, edges_1, width*height*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_edges_2, edges_2, width*height*sizeof(int), cudaMemcpyHostToDevice);

    // Allocate space on host for output arrays
    int *serial_output = (int *)malloc(width * height * sizeof(int));

    double serial_time_spent = serial_difference_filter(serial_output, edges_1, edges_2, width, height, threshold);

    struct timeval tv1, tv2;
    gettimeofday(&tv1, NULL);

    difference_filter<<<height, width>>>(dev_out, dev_edges_1, dev_edges_2, width, height, threshold);
    cudaMemcpy(output, dev_out, width * height * sizeof(int), cudaMemcpyDeviceToHost);

    gettimeofday(&tv2, NULL);
    double time_spent = (double)(tv2.tv_usec - tv1.tv_usec) / 1000000 + (double)(tv2.tv_sec - tv1.tv_sec);
    printf("Parallel difference filter execution time: %f seconds\n", time_spent);

    // Responsible programmer
    cudaFree(dev_out);
    cudaFree(dev_edges_1);
    cudaFree(dev_edges_2);

    for (int i = 0; i < width*height; i++) {
        if (serial_output[i] != output[i]) {
            printf("Error! Serial and parallel computation results are inconsistent: %d, %d\n", serial_output[i], output[i]);
        }
    }
    printf("Estimated parallelization speedup: %f\n", serial_time_spent/time_spent);
}

int main() {
    // Load external image into array
    Mat image = imread("../images/nvidia_1000_1000.jpg", 0);
    int *x = (int *)malloc(image.cols * image.rows * sizeof(int));
    int *out = (int *)malloc(100000000 * sizeof(int));
    for (int i = 0; i < image.rows; i++) {
        for (int j = 0; j < image.cols; j++) {
            x[i * image.rows + j] = image.at<uchar>(i, j);
        }
    }
    int threshold = 3;
    motion_track(out, x, x, image.rows, image.cols, threshold);
}
