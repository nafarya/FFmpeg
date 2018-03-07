/*
 * This file is part of FFmpeg.
 *
 * FFmpeg is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 *
 * FFmpeg is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with FFmpeg; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA
 */

__kernel void convolution_global(__write_only image2d_t dst,
                                 __read_only  image2d_t src,
                                 int coef_matrix_size,
                                 __constant float *coef_matrix,
                                 float div,
                                 float bias)
{
    const sampler_t sampler = (CLK_NORMALIZED_COORDS_FALSE | CLK_FILTER_NEAREST);

    const int half_matrix_size = (coef_matrix_size / 2);
    int2 loc = (int2)(get_global_id(0), get_global_id(1));
    float4 convPix = (float4)(0.0f, 0.0f, 0.0f, 0.0f);

    for (int conv_i = -half_matrix_size; conv_i <= half_matrix_size; conv_i++) {
        for (int conv_j = -half_matrix_size; conv_j <= half_matrix_size; conv_j++) {
            float4 px = read_imagef(src, sampler, loc + (int2)(conv_j, conv_i));
            convPix += px * coef_matrix[(conv_i+1)*coef_matrix_size+(conv_j+1)];
        }
     }
     write_imagef(dst, loc, convPix * div + bias);
}

__kernel void convolution_local(__write_only image2d_t dst,
                                 __read_only  image2d_t src,
                                 int coef_matrix_size,
                                 __constant float *coef_matrix,
                                 float div,
                                 float bias)
{
    const sampler_t sampler = (CLK_NORMALIZED_COORDS_FALSE |
                               CLK_ADDRESS_CLAMP_TO_EDGE |
                               CLK_FILTER_NEAREST);

    const int block_size = 16;

    const int block_x = get_group_id(0) * block_size;
    const int block_y = get_group_id(1) * block_size;
    const int local_x  =  get_local_id(0);
    const int local_y  =  get_local_id(1);
    const int half_matrix_size = (coef_matrix_size / 2);

    __local float4 B[16*3*16*3];

    for (int i = -half_matrix_size; i <= half_matrix_size; i++) {
            for (int j = -half_matrix_size; j <= half_matrix_size; j++) {
                    B[((i+1)*block_size+local_x) * (block_size*coef_matrix_size) + (j+1)*block_size+local_y] =
                        read_imagef(src, sampler, (int2)(block_x+local_y + (j*block_size), block_y+local_x + (i*block_size)));
            }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    float4 convPix = (float4)(0.0f, 0.0f, 0.0f, 0.0f);
    for (int conv_i = -half_matrix_size; conv_i <= half_matrix_size; conv_i++) {
                    for (int conv_j = -half_matrix_size; conv_j <= half_matrix_size; conv_j++) {
                            float4 px = B[(local_x+conv_i+block_size) * (block_size*coef_matrix_size) + local_y+conv_j+block_size];
                            convPix += px * coef_matrix[(conv_i+1)*coef_matrix_size+(conv_j+1)];
                    }
            }
    barrier(CLK_LOCAL_MEM_FENCE);
    write_imagef(dst, (int2)(block_x+local_y, block_y+local_x), convPix * div + bias);

}
