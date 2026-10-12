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
inline int host_mlplayer_count=0;


