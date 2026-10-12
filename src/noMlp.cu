#include "noMlp.h"
#include "globals.h"
#include <vector>
#include <iostream>

void addlayer(int node_in,int node_out){
    layerstruct l;
    l={0};
    l.node_in=node_in;
    l.node_out=node_out;
    host_layerdata.push_back(l);
    host_mlplayer_count++;
    host_weightbuffersize+= node_in*node_out;
    host_nodedatabuffersize+=node_out;
    host_biasbuffersize+=node_out;
}
void setoffsets(){
    for(int i=0;i<host_mlplayer_count;i++){
        int k= i-1;
        layerstruct& l=host_layerdata[i];

        l.weight_index=i==0?0: l.weight_index+l.node_in*l.node_out;
        l.nodedata_index= i==0?0:l.nodedata_index+l.node_out;
        l.bias_index=i==0?0: l.bias_index+ l.node_out;
    }
}

void nomlp::initlayers(std::vector<int> neurons){

    host_layerdata.clear();
    int n=neurons.size();
if(n==0)printf("neurons size cannot be zero \n");
    for(int i=0;i<n;i+=2){
        int in=i==0?neurons[i]: neurons[i-1];
        int out=neurons[i+1];
        addlayer(in,out);
    }
    setoffsets();
}


