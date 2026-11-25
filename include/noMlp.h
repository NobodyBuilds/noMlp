#pragma once

#include <vector>
#include <string_view>
#include <cstdio>
#include <cuda_runtime.h>

#ifndef debug
#define debug false
#endif

template <typename... Args>
inline void logerror(const char *format, Args... args)
{
    if constexpr (debug)
        std::printf(format, args...);
    else
        std::printf("nomlp: error\n");
}

inline bool geterror(const char *label, cudaError_t err)
{
    if (err != cudaSuccess)
    {
        if constexpr (debug)
            std::printf("nomlp: %s failed: %s\n", label, cudaGetErrorString(err));
        else
            std::printf("nomlp: CUDA error: %s\n", cudaGetErrorString(err));
        return false;
    }
    return true;
}

inline bool checkkernel(const char *label)
{
    if constexpr (debug)
        return geterror(label, cudaGetLastError());
    return true;
}

inline bool checkkernels(const char *label)
{
    if constexpr (debug)
        return true;
    return geterror(label, cudaGetLastError());
}
struct layerstruct
{
    int node_in;
    int node_out;
    int weight_index;
    int bias_index;
    int nodedata_index;
};
enum class outputmode
{
    linearMse,
    softmaxCrossEntrophy
};
enum class activationType{
    tanh,
    relu,
    leakyRelu
};
struct modelconfig
{
    int node_in = 0;
    int node_out = 0;
    float lr = 0.0f;
    int datasize = 0;
    outputmode output_mode = outputmode::linearMse;
    int batchsize = 1;
    activationType activation =activationType::tanh;

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
    std::vector<float> batchInputs;
    std::vector<float> batchTargets;
    int layers_count = 0;

    float *d_inputs = nullptr;
    float *d_target = nullptr;

    float *d_weights = nullptr;
    float *column_major_weights = nullptr;
    float *d_bias = nullptr;

    float *d_nodevals = nullptr;
    float *d_preact = nullptr;
    float *d_delta = nullptr;
    float2 *adam_weights = nullptr; // x= m,y=v;
    float2 *adam_bias = nullptr;    // x= m,y=v;

    int adam_step = 1;
    int weightbuffersize = 0;
    int biasbuffersize = 0;
    int nodedatabuffersize = 0;
    int activationInt = 1;

    modelconfig modeldata;

    void init(modelconfig &model);
    void close();

   
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

   
   
    std::vector<float> forward(
        const std::vector<float> &inputs,int batchsize=-1);

    std::vector<float> backward(
        const std::vector<float> &targets,
       int batchsize =-1,
        bool external_gradient = false);

    void step();

    void uploadCustomWeights(std::vector<float>& weights);
    void uploadCustomBiases(std::vector<float>& bias);
    void getWeights(std::vector<float>& constainer);
    void getBias(std::vector<float>& constainer);

    
};
