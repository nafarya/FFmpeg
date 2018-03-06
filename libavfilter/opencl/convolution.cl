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
    const int BLOCKSIZE = 8;
    const int factor = 1;
    const int MATRIXSIZE = coef_matrix_size;
    const int HALFMATRIXSZ = (MATRIXSIZE / 2);

    const int X = get_global_id(0);
    const int Y = get_global_id(1);

    const int BLOCK_X = X * BLOCKSIZE;
    const int BLOCK_Y = Y * BLOCKSIZE;

    int2 loc    = (int2)(X, Y);
    float4 convPix = (float4)(0.0f, 0.0f, 0.0f, 0.0f);
    for (int conv_i = -HALFMATRIXSZ; conv_i <= HALFMATRIXSZ; conv_i++) {
        for (int conv_j = -HALFMATRIXSZ; conv_j <= HALFMATRIXSZ; conv_j++) {
            float4 px = read_imagef(src, sampler, loc + (int2)(conv_j, conv_i));
            convPix += px * coef_matrix[(conv_i+1)*MATRIXSIZE+(conv_j+1)];
        }
     }
     write_imagef(dst, loc, convPix / div + bias);
}

__kernel void convolution_local(__write_only image2d_t dst,
                            __read_only  image2d_t src)
{
    const sampler_t sampler = (CLK_NORMALIZED_COORDS_FALSE |
                               CLK_ADDRESS_CLAMP_TO_EDGE |
                               CLK_FILTER_NEAREST);

    const int factor = 1;
    const int MATRIXSIZE = 3;
    const float matrix[9] = {1, 1, 1, 1, -7, 1, 1, 1, 1};


    const int WS = 8;

    const int X = get_group_id(0);
    const int Y = get_group_id(1);

    const int BLOCK_X = X * WS;
    const int BLOCK_Y = Y * WS;

    const int f  =  get_local_id(0);
    const int f1 =  get_local_id(1);

    const int HALFMATRIXSZ = (MATRIXSIZE / 2);


    __local float4 B[8*3*8*3];

    for (int i = -1; i <= 1; i++) {
            for (int j = -1; j <= 1; j++) {
                    B[((i+1)*WS+f) * (WS*3) +(j+1)*WS+f1] = read_imagef(src, sampler, (int2)(BLOCK_X+f1 + (j*WS), BLOCK_Y+f + (i*WS)));
            }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    float4 convPix = (float4)(0.0f, 0.0f, 0.0f, 0.0f);
    for (int conv_i = -HALFMATRIXSZ; conv_i <= HALFMATRIXSZ; conv_i++) {
                    for (int conv_j = -HALFMATRIXSZ; conv_j <= HALFMATRIXSZ; conv_j++) {
                            float4 px = B[(f+conv_i+WS) * (WS*3) + f1+conv_j+WS];
                            convPix += px * matrix[(conv_i+1)*MATRIXSIZE+(conv_j+1)];
                    }
            }
    barrier(CLK_LOCAL_MEM_FENCE);
    write_imagef(dst, (int2)(BLOCK_X+f1, BLOCK_Y+f), convPix / factor);

}
