#pragma once

#include <vector>
#include <string_view>
#include <cstdio>
#include <cuda_runtime.h>


inline bool geterror(const char *label, cudaError_t err)
{
    if (err != cudaSuccess)
    {
        std::printf("nomlp: %s failed: %s\n", label, cudaGetErrorString(err));
        return false;
    }
    return true;
}
struct layerstruct
{
    int node_in;
    int node_out;
    int weight_index;
    int bias_index;
    int nodedata_index;
};

struct modelconfig
{
    int node_in = 0;
    int node_out = 0;
    float lr = 0.0f;
    int datasize = 0;
    bool batching = false;

    int batchsize = 1;
    std::string_view activation = "tanh";
    const char *checkpoint_adrs = nullptr;

    std::vector<int> network;
};

class noMlp
{
private:
public:
    std::vector<layerstruct> layers;
    std::vector<float> host_outputs;
    std::vector<float> host_weights_array;
    std::vector<float> host_bias_array;
    std::vector<int> indices;
    int layers_count = 0;

    float *d_inputs = nullptr;
    float *d_target = nullptr;
    

    float *d_weights = nullptr;
    float* column_major_weights=nullptr;
    float *d_bias = nullptr;

    float *d_nodevals = nullptr;
    float *d_preact = nullptr;
    float *d_delta = nullptr;
    float2 *adam_weights = nullptr; // x= m,y=v;
    float2 *adam_bias = nullptr;    // x= m,y=v;
    float* loss=nullptr;
    int adam_step = 1;
    int weightbuffersize = 0;
    int biasbuffersize = 0;
    int nodedatabuffersize = 0;
    int activationInt = 1;

    modelconfig modeldata;

    void init(modelconfig &model);
    void close();

    std::vector<float> prediction(std::vector<float> &inputs);
    float backward(std::vector<float> &targets);

    float train_batch(std::vector<float> &inputs, std::vector<float> &targets);

    std::vector<float> inference_batch(std::vector<float> &inputs);

    void save_checkpoint();
    void load_checkpoint();

    void update_checkpoint_address(const char *path)
    {
        modeldata.checkpoint_adrs = path;
    }
    void update_learning_rate(float lr)
    {
        modeldata.lr = lr;
    }

};
