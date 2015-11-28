#ifndef SEPARABLE_CONVOLUTION_H
#define SEPARABLE_CONVOLUTION_H

void separable_convolve(int *output, int *x, int x_width, int x_height, int *horizontal_filter, int *vertical_filter, int kernel_size, double constant_scalar);

#endif
