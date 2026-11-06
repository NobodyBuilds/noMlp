#pragma once
#include "noMlp.h"
#include <vector>
#include <cuda_runtime.h>

inline bool allocatemlpmemory(noMlp &m)
{
    modelconfig &model = m.modeldata;
    if (!geterror("cudaMalloc device weights", cudaMalloc(&m.d_weights, m.weightbuffersize * sizeof(float)))) return false;
    if (!geterror("cudaMalloc device column-major weights", cudaMalloc(&m.column_major_weights, m.weightbuffersize * sizeof(float)))) return false;
    if (!geterror("cudaMalloc device bias", cudaMalloc(&m.d_bias, m.biasbuffersize * sizeof(float)))) return false;
    if (!geterror("cudaMalloc device node values", cudaMalloc(&m.d_nodevals, static_cast<size_t>(model.batchsize) * m.nodedatabuffersize * sizeof(float)))) return false;
    if (!geterror("cudaMalloc device preactivations", cudaMalloc(&m.d_preact, static_cast<size_t>(model.batchsize) * m.nodedatabuffersize * sizeof(float)))) return false;
    if (!geterror("cudaMalloc device deltas", cudaMalloc(&m.d_delta, static_cast<size_t>(model.batchsize) * m.nodedatabuffersize * sizeof(float)))) return false;
    const size_t input_capacity = static_cast<size_t>(model.batchsize) * model.node_in;
    const size_t target_capacity = model.batching
        ? static_cast<size_t>(model.batchsize) * model.node_out
        : static_cast<size_t>(model.node_out);
    if (!geterror("cudaMalloc device inputs", cudaMalloc(&m.d_inputs, input_capacity * sizeof(float)))) return false;
    if (!geterror("cudaMalloc device targets", cudaMalloc(&m.d_target, target_capacity * sizeof(float)))) return false;

    if (!geterror("cudaMemcpy weights to device", cudaMemcpy(m.d_weights, m.host_weights_array.data(), m.weightbuffersize * sizeof(float), cudaMemcpyHostToDevice))) return false;
    if (!geterror("cudaMemcpy bias to device", cudaMemcpy(m.d_bias, m.host_bias_array.data(), m.biasbuffersize * sizeof(float), cudaMemcpyHostToDevice))) return false;

    if (!geterror("cudaMallocManaged device loss", cudaMallocManaged(&m.loss, sizeof(float)))) return false;
    if (!geterror("cudaMemset device loss", cudaMemset(m.loss, 0, sizeof(float)))) return false;
    return true;
}
