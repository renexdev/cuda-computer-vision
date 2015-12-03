#include <stdio.h>
#include "../separable-convolution/separable_convolution.h"


void compare_naive_separable_convolution(int input_width, int input_height) {
    // Benchmark serial (CPU) performance of naive convolution versus separated convolution on a fixed filter and randomly-valued input
    int *out = (int *)malloc((input_width + 2) * (input_height + 2) * sizeof(int));
    int h[9] = {1, 2, 1, 2, 4, 2, 1, 2, 1};
    int h_horizontal[3] = {1, 2, 1};
    int h_vertical[3] = {1, 2, 1};
    int *x = (int *)malloc(input_width * input_height * sizeof(int));
    for (int i = 0; i < input_width * input_height; i++) {
        x[i] = rand();
    }
    struct timeval tv1, tv2;
    gettimeofday(&tv1, NULL);
    serial_naive_convolve(out, x, h, input_width, input_height, 3, 3);
    gettimeofday(&tv2, NULL);
    double naive_computation_time = (double) (tv2.tv_usec - tv1.tv_usec) / 1000000 + (double) (tv2.tv_sec - tv1.tv_sec);

    gettimeofday(&tv1, NULL);
    serial_separable_convolve(out, x, h_horizontal, h_vertical, input_width, input_height, 3, 3, 1.0)
    gettimeofday(&tv2, NULL);
    double separable_computation_time = (double) (tv2.tv_usec - tv1.tv_usec) / 1000000 + (double) (tv2.tv_sec - tv1.tv_sec);

    printf("Serial naive convolution execution time: %f", naive_computation_time);
    printf("Serial separable convolution execution time: %f", separable_computation_time);
}

void compare_separable_convolution_speedup(int input_width, int input_height) {
    // Benchmark parallel speedup of Gaussian filter - serial CPU separated convolution vs. parallel GPU separated convolution
    int *out = (int *)malloc((input_width + 2) * (input_height + 2) * sizeof(int));
    int h[9] = {1, 2, 1, 2, 4, 2, 1, 2, 1};
    int h_horizontal[3] = {1, 2, 1};
    int h_vertical[3] = {1, 2, 1};
    int *x = (int *)malloc(input_width * input_height * sizeof(int));
    for (int i = 0; i < input_width * input_height; i++) {
        x[i] = rand();
    }
    struct timeval tv1, tv2;
    gettimeofday(&tv1, NULL);
    separable_convolve(out, x, input_width, input_height, h_horizontal, h_vertical, 3, 1.0);
    gettimeofday(&tv2, NULL);
    double parallel_computation_time = (double) (tv2.tv_usec - tv1.tv_usec) / 1000000 + (double) (tv2.tv_sec - tv1.tv_sec);

    gettimeofday(&tv1, NULL);
    serial_separable_convolve(out, x, h_horizontal, h_vertical, input_width, input_height, 3, 3, 1.0)
    gettimeofday(&tv2, NULL);
    double serial_computation_time = (double) (tv2.tv_usec - tv1.tv_usec) / 1000000 + (double) (tv2.tv_sec - tv1.tv_sec);
    double estimated_speedup = serial_computation_time/parallel_computation_time;

    printf("Parallel separable convolution execution time: %f", parallel_computation_time);
    printf("Serial separable convolution execution time: %f", serial_computation_time);
    printf("Estimated parallelization speedup: %f", estimated_speedup);
}

int main() {
    compare_naive_separable_convolution();
    compare_separable_convolution_speedup();
    return 0;
}
