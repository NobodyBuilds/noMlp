#include <cuda_runtime.h>
#include "backpropkernels.cuh"
#include "adam.h"
#include "noMlp.h"

void noMlp::step(){


     float betat2 = 1.0f / (1.0f - powf(beta2, adam_step));
    float betat1 = 1.0f / (1.0f - powf(beta1, adam_step));
    const float4 beta = {beta1, beta2, betat1, betat2};
    for (int l = layers_count - 1; l >= 0; l--)
    {
        const int nin = layers[l].node_in;
        const int l1d = l == 0 ? 0 : layers[l - 1].nodedata_index;
        const int lw = layers[l].weight_index;
        const int d = layers[l].nodedata_index;
        const int lb = layers[l].bias_index;

        const dim3 biasThreads(32, 16);
        const int biasBlocks = (layers[l].node_out + biasThreads.x - 1) / biasThreads.x;
        const std::size_t biasShared =
            static_cast<std::size_t>(biasThreads.x) * biasThreads.y * sizeof(float);
        proposed_tunebias<<<biasBlocks, biasThreads, biasShared>>>(
            layers[l].node_out, d, lb, d_delta, d_bias,
            nodedatabuffersize, modeldata.batchsize, adam_bias, beta, modeldata.lr);

        const dim3 weightThreads(32, 32);
        const dim3 weightBlocks(
            (layers[l].node_out + weightThreads.x - 1) / weightThreads.x,
            (nin + weightThreads.y - 1) / weightThreads.y);
        proposed_tuneweights<<<weightBlocks, weightThreads>>>(
            layers[l].node_out, nin, l, lw, d, l1d,
            d_delta, d_inputs, d_weights, d_nodevals,
            nodedatabuffersize, modeldata.batchsize, adam_weights, beta, modeldata.lr);
        if (!checkkernel("launch batch weight-update kernel"))
            return ;
    }
    adam_step++;
    
}

