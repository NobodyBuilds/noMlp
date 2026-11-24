#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "noMlp.h"
#include "globals.h"
#include "activations.h"
#include "adam.h"
#include "backpropbatched.h"
#include "backpropkernels.cuh"

#include <cmath>

__global__ void convertr2c(float *rw, float *cw, int n, int l1w, int l_out, int lnin)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;
    // row = l1w + i * l1_out
    // col =  l1w + k * l1_nin + i
    int rows = lnin;
    int row = i % rows;
    int col = i / rows;

    cw[i] = rw[l1w + (row * l_out + col)];
}

__global__ void compute_delta(int n, int d, int l1_nout, int l1w, int l1_nin,
                              int l1d, bool outlayer, const float *__restrict__ nodedata, float *dDelta,
                              const float *__restrict__ dWeights, const float *__restrict__ dtarget,
                              const float *__restrict__ dpreact, int nodedatasize, int activation_type, bool external_grad)
{

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int b = blockIdx.y;
    if (i >= n)
        return;
    if (outlayer)
    {
        const int outputOffset = b * nodedatasize + d;
        const int targetOffset = b * n;
        dDelta[outputOffset + i] = external_grad ? dtarget[targetOffset + i] : nodedata[outputOffset + i] - dtarget[targetOffset + i];
    }
    else
    {
        float sum = 0.0f;
#pragma unroll 4
        for (int k = 0; k < l1_nout; k++)
        {
            sum += dWeights[k * l1_nin + i] * dDelta[b * nodedatasize + l1d + k];
        }
        dDelta[b * nodedatasize + d + i] = sum * dslope(dpreact[b * nodedatasize + d + i], activation_type);
    }
}

__global__ void proposed_tunebias(
    int n, int d, int lb, const float *dDelta, float *dBias,
    int nodedatasize, int curbatch, float2 *adam_bias, float4 beta, float d_lr)
{
    extern __shared__ float smem[];
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = threadIdx.y;

    float sum = 0.0f;
    if (i < n)
    {
        for (int b = y; b < curbatch; b += blockDim.y)
            sum += dDelta[b * nodedatasize + d + i];
        smem[y * blockDim.x + threadIdx.x] = sum;
    }

    __syncthreads();

    if (y == 0 && i < n)
    {
        float biasgrad = 0.0f;
#pragma unroll 8
        for (int k = 0; k < blockDim.y; ++k)
            biasgrad += smem[k * blockDim.x + threadIdx.x];

        adam_update(dBias[lb + i], adam_bias[lb + i], biasgrad / curbatch,
                    d_lr, beta);
    }
}

__global__ void proposed_tuneweights(
    int n, int nin, int l, int lw, int d, int l1d,
    const float *dDelta, const float *ddata, float *dWeights,
    const float *dNodeData, int nodedatasize, int curbatch,
    float2 *adam_weights, float4 beta, float d_lr)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int k = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= n || k >= nin)
        return;

    float wgrad = 0.0f;
    for (int b = 0; b < curbatch; ++b)
    {
        const int offset = b * nodedatasize;
        const float nodeValue = (l == 0)
                                    ? ddata[b * nin + k]
                                    : dNodeData[offset + l1d + k];
        wgrad += dDelta[offset + d + i] * nodeValue;
    }

    const int weightIndex = lw + k * n + i;
    adam_update(dWeights[weightIndex], adam_weights[weightIndex],
                wgrad / curbatch, d_lr, beta);
}

bool backpropogationbatch(int cb, noMlp &m)
{

    if (cb <= 0 || m.layers.empty() || m.d_delta == nullptr ||
        m.d_weights == nullptr || m.d_bias == nullptr || m.d_target == nullptr ||
        m.d_inputs == nullptr || m.d_nodevals == nullptr || m.d_preact == nullptr ||
        m.adam_weights == nullptr || m.adam_bias == nullptr)
    {
        logerror("nomlp: invalid model state for batch backpropagation\n");
        return false;
    }

    int threads = 256;
    for (int l = m.layers_count - 1; l >= 0; l--)
    {

        int d = m.layers[l].nodedata_index;
        int n = m.layers[l].node_out;
        int l1out = 0, l1w = 0, l1d = 0, l1nin = 0;

        if (l < m.layers_count - 1)
        {
            l1out = m.layers[l + 1].node_out;
            l1w = m.layers[l + 1].weight_index;
            l1d = m.layers[l + 1].nodedata_index;
            l1nin = m.layers[l + 1].node_in;
        }

        bool outlayer = (l == m.layers_count - 1) ? 1 : 0;

        if (!outlayer)
        {

            size_t layer_weight_size = l1nin * l1out;
            convertr2c<<<(layer_weight_size + 255) / 256, 256>>>(m.d_weights, m.column_major_weights, layer_weight_size, m.layers[l + 1].weight_index, m.layers[l + 1].node_out, l1nin);
        }
        int blocks = (n + threads - 1) / threads;
        dim3 deltaGrid(blocks, cb);
        compute_delta<<<deltaGrid, threads>>>(n, d, l1out, l1w, l1nin, l1d, outlayer, m.d_nodevals, m.d_delta, m.column_major_weights, m.d_target, m.d_preact, m.nodedatabuffersize, m.activationInt, false);
        if (!checkkernel("launch batch delta kernel"))
            return false;
    }

    float betat2 = 1.0f / (1.0f - powf(beta2, m.adam_step));
    float betat1 = 1.0f / (1.0f - powf(beta1, m.adam_step));
    const float4 beta = {beta1, beta2, betat1, betat2};
    for (int l = m.layers_count - 1; l >= 0; l--)
    {
        const int nin = m.layers[l].node_in;
        const int l1d = l == 0 ? 0 : m.layers[l - 1].nodedata_index;
        const int lw = m.layers[l].weight_index;
        const int d = m.layers[l].nodedata_index;
        const int lb = m.layers[l].bias_index;

        const dim3 biasThreads(32, 16);
        const int biasBlocks = (m.layers[l].node_out + biasThreads.x - 1) / biasThreads.x;
        const std::size_t biasShared =
            static_cast<std::size_t>(biasThreads.x) * biasThreads.y * sizeof(float);
        proposed_tunebias<<<biasBlocks, biasThreads, biasShared>>>(
            m.layers[l].node_out, d, lb, m.d_delta, m.d_bias,
            m.nodedatabuffersize, cb, m.adam_bias, beta, m.modeldata.lr);

        const dim3 weightThreads(32, 32);
        const dim3 weightBlocks(
            (m.layers[l].node_out + weightThreads.x - 1) / weightThreads.x,
            (nin + weightThreads.y - 1) / weightThreads.y);
        proposed_tuneweights<<<weightBlocks, weightThreads>>>(
            m.layers[l].node_out, nin, l, lw, d, l1d,
            m.d_delta, m.d_inputs, m.d_weights, m.d_nodevals,
            m.nodedatabuffersize, cb, m.adam_weights, beta, m.modeldata.lr);
        if (!checkkernel("launch batch weight-update kernel"))
            return false;
    }
    m.adam_step++;
    return true;
}
