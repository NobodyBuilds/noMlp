#pragma once
#include <iostream>
#include <vector>
#include"noMlp.h"
#include <cuda.h>
#include <cuda_runtime.h>

inline std::vector<float> host_weights_array;
inline std::vector<float> host_bias_array;
inline std::vector<float> host_outputs;
inline std::vector<int> indices;

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



inline float* device_mlp_inputs=nullptr;
inline float* device_mlp_target=nullptr;
inline int* device_indices=nullptr;
inline float* device__weights=nullptr;
inline float* device__bias=nullptr;
inline float* device__nodevals=nullptr;
inline float* device__preact=nullptr;
inline float* device__delta=nullptr;
//
inline  int activationInt; //1= tanh,2= relu,3=leakyrelu
inline __device__ int dactivationint=1;
inline __device__ float d_lr=0.0f;

//
//helpers
inline int blocks(int n,int thread){



    return (n + thread- 1) / thread;
}


inline modelst model;

inline __device__ float activation(float x ){
    int type =dactivationint;
  
    switch (type){
       case 1: // Tanh
        return tanhf(x);

    case 2: // ReLU
        return x > 0.0f ? x : 0.0f;

    case 3: // Leaky ReLU
        return x > 0.0f ? x : 0.01f * x;

    default:
        return x; 
    }
}

inline __device__ float dslope(float x)
{
	switch (dactivationint)
	{
	case 1: // Tanh
	{
		const float t = tanhf(x);
		return 1.0f - t * t;
	}
	case 2: // ReLU
		return x > 0.0f ? 1.0f : 0.0f;
	case 3: // LeakyReLU
		return x > 0.0f ? 1.0f : 0.01f;
	default:
		return 1.0f;
	}
}

void backpropogationbatch(int cb);
