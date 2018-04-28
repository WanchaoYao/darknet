#include "im2col.h"
#include <stdio.h>

// 获取图像像素值
float im2col_get_pixel(float *im, int height, int width, int channels,
                        int row, int col, int channel, int pad)
{
    row -= pad;
    col -= pad;

    // pad 部分填充 0
    if (row < 0 || col < 0 ||
        row >= height || col >= width) return 0;

    return im[col + width*(row + height*channel)];
}

// 将图像每一个kernel转换成一列
//From Berkeley Vision's Caffe!
//https://github.com/BVLC/caffe/blob/master/LICENSE
void im2col_cpu(float* data_im,
     int channels,  int height,  int width,
     int ksize,  int stride, int pad, float* data_col) 
{
    int c,h,w;

    // 计算 kernel 的个数 = height_col * width_col
    int height_col = (height + 2*pad - ksize) / stride + 1;
    int width_col = (width + 2*pad - ksize) / stride + 1;

    // 计算一个 kernel 的大小
    int channels_col = channels * ksize * ksize;

    // 将每一个 kernel 大小的图像转换成 一列
    for (c = 0; c < channels_col; ++c) {
        // 一个 kernel 上的坐标 h_offset，w_offset，c_im
        int w_offset = c % ksize;
        int h_offset = (c / ksize) % ksize;
        int c_im = c / ksize / ksize;

        // 遍历所有 kernel
        for (h = 0; h < height_col; ++h) {
            for (w = 0; w < width_col; ++w) {
                // 一个 kernel 的像素点对应到图像上的坐标 im_row，im_col，c_im
                int im_row = h_offset + h * stride;
                int im_col = w_offset + w * stride;

                // 第 col_index 列(kernel)
                int col_index = (c * height_col + h) * width_col + w;
                data_col[col_index] = im2col_get_pixel(data_im, height, width, channels,
                        im_row, im_col, c_im, pad);
            }
        }
    }
}

