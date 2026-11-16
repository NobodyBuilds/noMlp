#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "softmax.cuh"

__global__ void d_softmax(int n, int didx, int nds, float *data)
{
    extern __shared__ float reduction[];

    const int sample = blockIdx.x;
    const int tid = threadIdx.x;
    const int sample_offset = sample * nds + didx;

  
    float local_max = -FLT_MAX;
    for (int i = tid; i < n; i += blockDim.x)
        local_max = fmaxf(local_max, data[sample_offset + i]);

    reduction[tid] = local_max;
    __syncthreads();

    
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1)
    {
        if (tid < stride)
            reduction[tid] = fmaxf(reduction[tid], reduction[tid + stride]);
        __syncthreads();
    }
    const float max_value = reduction[0];

    
    float local_sum = 0.0f;
    for (int i = tid; i < n; i += blockDim.x)
        local_sum += expf(data[sample_offset + i] - max_value);

    reduction[tid] = local_sum;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1)
    {
        if (tid < stride)
            reduction[tid] += reduction[tid + stride];
        __syncthreads();
    }
    const float denominator = reduction[0];

   
    for (int i = tid; i < n; i += blockDim.x)
    {
        const int index = sample_offset + i;
        data[index] = expf(data[index] - max_value) / denominator;
    }
}

void applySoftmax(noMlp &m, int curbatch)
{
    const int threads = 256;
    const int n = m.modeldata.node_out;
    const size_t smem = static_cast<size_t>(threads) * sizeof(float);
    d_softmax<<<curbatch, threads, smem>>>(
        n,
        m.layers[m.layers_count - 1].nodedata_index,
        m.nodedatabuffersize,
        m.d_nodevals);
}