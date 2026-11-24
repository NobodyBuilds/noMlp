#include <cuda_runtime.h>
#include "noMlp.h"

void noMlp::uploadCustomWeights(std::vector<float> &weights)
{

    if (weights.size() != weightbuffersize)
    {
        printf("custom weights does not match current network structre \n");
        return;
    }

    if (!geterror("uploading custom weights", cudaMemcpy(d_weights, weights.data(), weightbuffersize * sizeof(float), cudaMemcpyHostToDevice)))
    {
        return;
    }
}
void noMlp::uploadCustomBiases(std::vector<float> &bias)
{

    if (bias.size() != biasbuffersize)
    {
        printf("custom bias does not match current network structre \n");
        return;
    }

    if (!geterror("uploading custom bias", cudaMemcpy(d_bias, bias.data(), biasbuffersize * sizeof(float), cudaMemcpyHostToDevice)))
    {
        return;
    }
}

void noMlp::getWeights(std::vector<float>& constainer){
    constainer.resize(weightbuffersize);
    if (!geterror("getting weights copy", cudaMemcpy(&constainer, d_weights, weightbuffersize * sizeof(float), cudaMemcpyDeviceToHost)))
    {
        return;
    }

}
void noMlp::getBias(std::vector<float>& constainer){
    constainer.resize(biasbuffersize);
    if (!geterror("getting bias copy", cudaMemcpy(&constainer, d_bias, biasbuffersize * sizeof(float), cudaMemcpyDeviceToHost)))
    {
        return;
    }

}
