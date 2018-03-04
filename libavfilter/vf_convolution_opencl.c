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

#include "avfilter.h"
#include "internal.h"
#include "opencl.h"
#include "opencl_source.h"
#include "video.h"

#define MAX_DIAMETER 23

typedef struct ConvolutionOpenCLContext {
    OpenCLFilterContext ocf;

    int              initialised;
    cl_kernel        kernel;
    cl_command_queue command_queue;

    cl_int   size_x;
    cl_int   size_y;

    int global;

} ConvolutionOpenCLContext;


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
    ctx->global = (ctx->size_x  > ctx->size_y);

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
    float *matrix;
    double val, sum;
    cl_int cle;
    cl_mem buffer;
    size_t matrix_bytes;
    float diam_x, diam_y, amount;
    int err, p, x, y, size_x, size_y;

    return 0;
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

        if (ctx->global) {
            global_work[0] = output->width;
            global_work[1] = output->height;
        } else {
            global_work[0] = FFALIGN(output->width,  16);
            global_work[1] = FFALIGN(output->height, 16);
            local_work[0]  = 16;
            local_work[1]  = 16;
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
