#pragma once
#include <iostream>
#include <vector>
#include"noMlp.h"
#include <cuda.h>
#include <cuda_runtime.h>

inline std::vector<float> host_weights_array;
inline std::vector<float> host_bias_array;
inline std::vector<float> host_nodedata_array;

inline int host_weightbuffersize=0;
inline int host_biasbuffersize=0;
inline int host_nodedatabuffersize=0;

struct layerstruct{
    int node_in;
    int node_out;
    int weight_index;
    int bias_index;
    int nodedata_index;
};

inline std::vector<layerstruct> host_layerdata;
inline int host_mlp_layer_count=0;

struct modelst{
    float lr;
    int batchsize;
    int activation; //1= tanh,2= relu,3=leakyrelu
};

inline float* device_mlp_inputs=nullptr;
inline float* device_mlp_targets=nullptr;
inline int* device_indices=nullptr;
inline float* device__weights=nullptr;
inline float* device__bias=nullptr;
inline float* device__nodevals=nullptr;
inline float* device__preact=nullptr;
inline float* device__delta=nullptr;


inline int blocks(int n,int thread){
    return (n + thread- 1) / thread;
}