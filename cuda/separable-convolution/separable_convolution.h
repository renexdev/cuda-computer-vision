#ifndef SEPARABLE_CONVOLUTION_H
#define SEPARABLE_CONVOLUTION_H

void separable_convolve(int *output, int *x, int x_width, int x_height, int *horizontal_filter, int *vertical_filter, int kernel_size, double constant_scalar);

double serial_separable_convolve(int *out, int *x, int *horizontal_filter, int *vertical_filter, int x_width, int x_height, int horizontal_filter_width, int vertical_filter_height, double constant_scalar);

double serial_naive_convolve(int *out, int *x, int *h, int x_width, int x_height, int h_width, int h_height);

#endif
