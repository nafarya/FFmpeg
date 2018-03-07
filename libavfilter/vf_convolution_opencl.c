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

#include "libavutil/common.h"
#include "libavutil/imgutils.h"
#include "libavutil/mem.h"
#include "libavutil/opt.h"
#include "libavutil/pixdesc.h"
#include "libavutil/avstring.h"


#include "avfilter.h"
#include "internal.h"
#include "opencl.h"
#include "opencl_source.h"
#include "video.h"

typedef struct ConvolutionOpenCLContext {
    OpenCLFilterContext ocf;

    int              initialised;
    cl_kernel        kernel;
    cl_command_queue command_queue;

    cl_int   size_x;
    cl_int   size_y;

    char *matrix_str;
    cl_int size;

    cl_int matrix_length;
    cl_float rdiv;
    cl_float bias;
    cl_mem matrix;

    int global;

} ConvolutionOpenCLContext;


const float default3x3[9] = {0, 0, 0,
                             0, 1, 0,
                             0, 0, 0};

const float default5x5[25] = {0, 0, 0, 0, 0,
                              0, 0, 0, 0, 0,
                              0, 0, 1, 0, 0,
                              0, 0, 0, 0, 0,
                              0, 0, 0, 0, 0};

const float default7x7[49] = {0, 0, 0, 0, 0, 0, 0,
                              0, 0, 0, 0, 0, 0, 0,
                              0, 0, 0, 0, 0, 0, 0,
                              0, 0, 0, 1, 0, 0, 0,
                              0, 0, 0, 0, 0, 0, 0,
                              0, 0, 0, 0, 0, 0, 0,
                              0, 0, 0, 0, 0, 0, 0};

static int convolution_opencl_init(AVFilterContext *avctx)
{
    ConvolutionOpenCLContext *ctx = avctx->priv;
    cl_int cle;
    int err;

    err = ff_opencl_filter_load_program(avctx, &ff_opencl_source_convolution, 1);
    if (err < 0)
        goto fail;

    ctx->command_queue = clCreateCommandQueue(ctx->ocf.hwctx->context,
                                              ctx->ocf.hwctx->device_id,
                                              0, &cle);
    if (!ctx->command_queue) {
        av_log(avctx, AV_LOG_ERROR, "Failed to create OpenCL "
               "command queue: %d.\n", cle);
        err = AVERROR(EIO);
        goto fail;
    }

    // Use global kernel if mask size will be too big for the local store..
    //ctx->global = (ctx->size_x  > ctx->size_y);
    ctx->global = 1;

    ctx->kernel = clCreateKernel(ctx->ocf.program,
                                 ctx->global ? "convolution_global"
                                             : "convolution_local", &cle);
    if (!ctx->kernel) {
        av_log(avctx, AV_LOG_ERROR, "Failed to create kernel: %d.\n", cle);
        err = AVERROR(EIO);
        goto fail;
    }

    ctx->initialised = 1;
    return 0;

fail:
    if (ctx->command_queue)
        clReleaseCommandQueue(ctx->command_queue);
    if (ctx->kernel)
        clReleaseKernel(ctx->kernel);
    return err;
}



static int convolution_opencl_make_filter_params(AVFilterContext *avctx)
{
    ConvolutionOpenCLContext *ctx = avctx->priv;
    const AVPixFmtDescriptor *desc;



    char *p, *arg, *saveptr = NULL;

    float input_matrix[49];
    p = ctx->matrix_str;
    while (ctx->matrix_length < 49) {
        if (!(arg = av_strtok(p, " ", &saveptr)))
            break;
        p = NULL;
        sscanf(arg, "%f", &input_matrix[ctx->matrix_length]);
        ctx->matrix_length++;
    }

    float *matrix;
    size_t matrix_bytes = sizeof(float)*ctx->matrix_length;
    matrix = av_malloc(matrix_bytes);


    if (ctx->matrix_length == 9) {
        ctx->size = 3;
        memcpy(matrix, default3x3, ctx->matrix_length);
    } else if (ctx->matrix_length == 25) {
            ctx->size = 5;
            memcpy(matrix, default5x5, ctx->matrix_length);
    } else if (ctx->matrix_length == 49) {
            ctx->size = 7;
            memcpy(matrix, default7x7, ctx->matrix_length);
    } else {
        return AVERROR(EINVAL);
    }

    for (int i = 0; i < ctx->matrix_length; i++)
        matrix[i] = input_matrix[i];


    cl_int cle;
    cl_mem buffer;
    int err = 0;

    buffer = clCreateBuffer(ctx->ocf.hwctx->context,
                            CL_MEM_READ_ONLY |
                            CL_MEM_COPY_HOST_PTR |
                            CL_MEM_HOST_NO_ACCESS,
                            matrix_bytes, matrix, &cle);
    if (!buffer) {
        av_log(avctx, AV_LOG_ERROR, "Failed to create matrix buffer: "
               "%d.\n", cle);
        err = AVERROR(EIO);
        goto fail;
    }
    ctx->matrix = buffer;

    return err;
fail:
    av_freep(&matrix);
    return err;


}

static int convolution_opencl_filter_frame(AVFilterLink *inlink, AVFrame *input)
{
    AVFilterContext    *avctx = inlink->dst;
    AVFilterLink     *outlink = avctx->outputs[0];
    ConvolutionOpenCLContext *ctx = avctx->priv;
    AVFrame *output = NULL;
    cl_int cle;
    size_t global_work[2];
    size_t local_work[2];
    cl_mem src, dst;
    int err, p;

    av_log(ctx, AV_LOG_DEBUG, "Filter input: %s, %ux%u (%"PRId64").\n",
           av_get_pix_fmt_name(input->format),
           input->width, input->height, input->pts);

    if (!input->hw_frames_ctx)
        return AVERROR(EINVAL);

    if (!ctx->initialised) {
        err = convolution_opencl_init(avctx);
        if (err < 0)
            goto fail;

        err = convolution_opencl_make_filter_params(avctx);
        if (err < 0)
            goto fail;
    }

    output = ff_get_video_buffer(outlink, outlink->w, outlink->h);
    if (!output) {
        err = AVERROR(ENOMEM);
        goto fail;
    }

    for (p = 0; p < FF_ARRAY_ELEMS(output->data); p++) {
        src = (cl_mem) input->data[p];
        dst = (cl_mem)output->data[p];


        if (!dst)
            break;

        cle = clSetKernelArg(ctx->kernel, 0, sizeof(cl_mem), &dst);
        if (cle != CL_SUCCESS) {
            av_log(avctx, AV_LOG_ERROR, "Failed to set kernel "
                   "destination image argument: %d.\n", cle);
            goto fail;
        }
        cle = clSetKernelArg(ctx->kernel, 1, sizeof(cl_mem), &src);
        if (cle != CL_SUCCESS) {
            av_log(avctx, AV_LOG_ERROR, "Failed to set kernel "
                   "source image argument: %d.\n", cle);
            goto fail;
        }
        cle = clSetKernelArg(ctx->kernel, 2, sizeof(cl_int), &ctx->size);
        if (cle != CL_SUCCESS) {
            av_log(avctx, AV_LOG_ERROR, "Failed to set kernel "
                   "matrix size argument: %d.\n", cle);
            goto fail;
        }
        cle = clSetKernelArg(ctx->kernel, 3, sizeof(cl_mem), &ctx->matrix);
        if (cle != CL_SUCCESS) {
            av_log(avctx, AV_LOG_ERROR, "Failed to set kernel "
                   "matrix argument: %d.\n", cle);
            goto fail;
        }
        cle = clSetKernelArg(ctx->kernel, 4, sizeof(cl_float), &ctx->rdiv);
        if (cle != CL_SUCCESS) {
            av_log(avctx, AV_LOG_ERROR, "Failed to set kernel "
                   "div argument: %d.\n", cle);
            goto fail;
        }
        cle = clSetKernelArg(ctx->kernel, 5, sizeof(cl_float), &ctx->bias);
        if (cle != CL_SUCCESS) {
            av_log(avctx, AV_LOG_ERROR, "Failed to set kernel "
                   "bias argument: %d.\n", cle);
            goto fail;
        }


        if (ctx->global) {
            global_work[0] = output->width;
            global_work[1] = output->height;
        } else {
            global_work[0] = FFALIGN(output->width,  8);
            global_work[1] = FFALIGN(output->height, 8);
            local_work[0]  = 8;
            local_work[1]  = 8;
        }

        av_log(avctx, AV_LOG_DEBUG, "Run kernel on plane %d "
               "(%"SIZE_SPECIFIER"x%"SIZE_SPECIFIER").\n",
               p, global_work[0], global_work[1]);

        cle = clEnqueueNDRangeKernel(ctx->command_queue, ctx->kernel, 2, NULL,
                                     global_work, ctx->global ? NULL : local_work,
                                     0, NULL, NULL);
        if (cle != CL_SUCCESS) {
            av_log(avctx, AV_LOG_ERROR, "Failed to enqueue kernel: %d.\n",
                   cle);
            err = AVERROR(EIO);
            goto fail;
        }
    }

    cle = clFinish(ctx->command_queue);
    if (cle != CL_SUCCESS) {
        av_log(avctx, AV_LOG_ERROR, "Failed to finish command queue: %d.\n",
               cle);
        err = AVERROR(EIO);
        goto fail;
    }

    err = av_frame_copy_props(output, input);
    if (err < 0)
        goto fail;

    av_frame_free(&input);

    av_log(ctx, AV_LOG_DEBUG, "Filter output: %s, %ux%u (%"PRId64").\n",
           av_get_pix_fmt_name(output->format),
           output->width, output->height, output->pts);

    return ff_filter_frame(outlink, output);

fail:
    clFinish(ctx->command_queue);
    av_frame_free(&input);
    av_frame_free(&output);
    return err;
}

static av_cold void convolution_opencl_uninit(AVFilterContext *avctx)
{
    ConvolutionOpenCLContext *ctx = avctx->priv;
    cl_int cle;
    int i;


    if (ctx->kernel) {
        cle = clReleaseKernel(ctx->kernel);
        if (cle != CL_SUCCESS)
            av_log(avctx, AV_LOG_ERROR, "Failed to release "
                   "kernel: %d.\n", cle);
    }

    if (ctx->command_queue) {
        cle = clReleaseCommandQueue(ctx->command_queue);
        if (cle != CL_SUCCESS)
            av_log(avctx, AV_LOG_ERROR, "Failed to release "
                   "command queue: %d.\n", cle);
    }

    ff_opencl_filter_uninit(avctx);
}

#define OFFSET(x) offsetof(ConvolutionOpenCLContext, x)
#define FLAGS (AV_OPT_FLAG_FILTERING_PARAM | AV_OPT_FLAG_VIDEO_PARAM)
static const AVOption convolution_opencl_options[] = {
    { "m",    "set matrix ",  OFFSET(matrix_str), AV_OPT_TYPE_STRING, {.str="0 0 0 0 1 0 0 0 0"}, 0, 0, FLAGS },
    { "rdiv", "set rdiv",     OFFSET(rdiv),      AV_OPT_TYPE_FLOAT,  {.dbl=1.0}, 0.0, INT_MAX, FLAGS},
    { "bias", "set bias",     OFFSET(bias),     AV_OPT_TYPE_FLOAT,  {.dbl=0.0}, 0.0, INT_MAX, FLAGS},

    { NULL }
};

AVFILTER_DEFINE_CLASS(convolution_opencl);

static const AVFilterPad convolution_opencl_inputs[] = {
    {
        .name         = "default",
        .type         = AVMEDIA_TYPE_VIDEO,
        .filter_frame = &convolution_opencl_filter_frame,
        .config_props = &ff_opencl_filter_config_input,
    },
    { NULL }
};

static const AVFilterPad convolution_opencl_outputs[] = {
    {
        .name         = "default",
        .type         = AVMEDIA_TYPE_VIDEO,
        .config_props = &ff_opencl_filter_config_output,
    },
    { NULL }
};

AVFilter ff_vf_convolution_opencl = {
    .name           = "convolution_opencl",
    .description    = NULL_IF_CONFIG_SMALL("Apply convolution mask to input video"),
    .priv_size      = sizeof(ConvolutionOpenCLContext),
    .priv_class     = &convolution_opencl_class,
    .init           = &ff_opencl_filter_init,
    .uninit         = &convolution_opencl_uninit,
    .query_formats  = &ff_opencl_filter_query_formats,
    .inputs         = convolution_opencl_inputs,
    .outputs        = convolution_opencl_outputs,
    .flags_internal = FF_FILTER_FLAG_HWFRAME_AWARE,
};
