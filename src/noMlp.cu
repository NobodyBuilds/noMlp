#include "noMlp.h"
#include "globals.h"
#include <vector>
#include <iostream>
#include <random>
#include <fstream>
#include <string>
#include <cstdio>
#include <string_view>
void addlayer(int node_in,int node_out){
    layerstruct l;
    l={0};
    l.node_in=node_in;
    l.node_out=node_out;
    host_layerdata.push_back(l);
    host_mlp_layer_count++;
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
void initWB(){
    host_weights_array.resize(host_weightbuffersize);
    host_bias_array.resize(host_biasbuffersize);
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-0.05f,0.05f);

    for(int i=0;i<host_weightbuffersize;i++){
        host_weights_array[i] =dist(rng);
    }
    for(int i=0;i<host_biasbuffersize;i++){
        host_bias_array[i] =dist(rng);
    }

}

void nomlp::initlizeWeights(){
    initWB();
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




void nomlp::init(float lr,int batchsize,std::string activation ){
    noMlp.model.lr=lr;
    noMlp.model.batchsize=batchsize;

    if(activation =="tanh")model.activation=1;
    else if(activation =="relu")model.activation=2;
    if(activation =="leakyrelu")model.activation=3;
    
    cudaError_t err;
    err= cudaMalloc(&device__weights,host_weightbuffersize *sizeof(float)); 
    if(err!=cudaSuccess){
        printf("device_weights memory allocation failed %s",cudaGetErrorString(err));    
    }
    err= cudaMalloc(&device__bias,host_biasbuffersize *sizeof(float)); 
    if(err!=cudaSuccess){
        printf("device_bias memory allocation failed %s",cudaGetErrorString(err));    
    }
    err= cudaMalloc(&device__nodevals,host_nodedatabuffersize *sizeof(float)); 
    if(err!=cudaSuccess){
        printf("device_nodevals memory allocation failed %s",cudaGetErrorString(err));    
    }
    err= cudaMalloc(&device__preact,host_nodedatabuffersize *sizeof(float)); 
    if(err!=cudaSuccess){

        printf("device_preactivation memory allocation failed %s",cudaGetErrorString(err));    
    }
    err= cudaMalloc(&device__delta,host_nodedatabuffersize *sizeof(float)); 
    if(err!=cudaSuccess){
        printf("device_dfelta memory allocation failed %s",cudaGetErrorString(err));    
    }
    err= cudaMalloc(&device_indices,batchsize *sizeof(int)); 
    if(err!=cudaSuccess){
        printf("device_indices memory allocation failed %s",cudaGetErrorString(err));    
    }
    
}

void nomlp::loadWeights(const char* weights,const char* bias){
     host_weights_array.resize(host_weightbuffersize);
    host_bias_array.resize(host_biasbuffersize);


    std::fstream a(weights,std::ios::in);
    if(!a.is_open()){
        printf("error opening %s file",weights);
    }
    std::string w;
    int wi=0;
    while(a>>w){
        host_weights_array[wi++]=std::stof(w);
    }
    a.close();
    if(wi!=host_weightbuffersize){
        printf(" weights file count mismatch ,check layers neurons or delete obselete weight file");
        return;
    }

    std::fstream b( bias,std::ios::in);
     if(!b.is_open()){
        printf("error opening %s file",bias);

    }
    std::string z;
    int bi=0;
    while(b>>z){
        host_bias_array[bi++]=std::stof(z);
    }
    b.close();
if(bi!=host_biasbuffersize){
        printf(" bias file count mismatch ,check layers neurons or delete obselete weight file");
        return;
    }

}