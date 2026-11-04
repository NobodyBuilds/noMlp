#pragma once 

#include "noMlp.h"

inline void addlayer(noMlp& m,int node_in,int node_out){
    layerstruct l;
    l={0};
    l.node_in=node_in;
    l.node_out=node_out;
    m.layers.push_back(l);
    m.layers_count++;
    m.weightbuffersize+= node_in*node_out;
    m.nodedatabuffersize+=node_out;
    m.biasbuffersize+=node_out;
}
inline void setoffsets(noMlp& m){
    int weight_offset = 0;
    int node_offset = 0;
    int bias_offset = 0;

    for (int i=0 ;i<m.layers_count;i++) {
        layerstruct& layer = m.layers[i];
        layer.weight_index = weight_offset;
        layer.nodedata_index = node_offset;
        layer.bias_index = bias_offset;

        weight_offset += layer.node_in * layer.node_out;
        node_offset += layer.node_out;
        bias_offset += layer.node_out;
    }
}

inline bool initlayers(const std::vector<int>& neurons,noMlp& m){
    if (neurons.size() < 2) {
        std::printf("nomlp: network must contain at least input and output sizes\n");
        return false;
    }

    for (int node_count : neurons) {
        if (node_count <= 0) {
            std::printf("nomlp: network layer sizes must be positive\n");
            return false;
        }
    }

    m.layers.clear();
    m.layers_count = 0;
    m.weightbuffersize = 0;
    m.biasbuffersize = 0;
    m.nodedatabuffersize = 0;

    for (size_t i = 0; i + 1 < neurons.size(); ++i) {
        addlayer(m,neurons[i], neurons[i + 1]);
    }

    setoffsets(m);

    return true;
}


