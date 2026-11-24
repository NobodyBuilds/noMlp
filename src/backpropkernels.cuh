#pragma once
#include <cuda_runtime.h>

__global__ void compute_delta(int n, int d, int l1_nout, int l1w, int l1_nin,
                              int l1d, bool outlayer, const float *__restrict__ nodedata, float *dDelta,
                              const float *__restrict__ dWeights, const float *__restrict__ dtarget,
                              const float *__restrict__ dpreact, int nodedatasize, int activation_type,bool external_grad);

__global__ void convertr2c(float *rw, float *cw, int n, int l1w, int l_out, int lnin);

__global__ void proposed_tunebias(
    int n, int d, int lb, const float *dDelta, float *dBias,
    int nodedatasize, int curbatch, float2 *adam_bias, float4 beta, float d_lr);

__global__ void proposed_tuneweights(
    int n, int nin, int l, int lw, int d, int l1d,
    const float *dDelta, const float *ddata, float *dWeights,
    const float *dNodeData, int nodedatasize, int curbatch,
    float2 *adam_weights, float4 beta, float d_lr);