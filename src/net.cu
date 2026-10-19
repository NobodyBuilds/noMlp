#include "noMlp.h"
#include "globals.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>
#include <random>
#include <algorithm>


__global__ void firstlayer(const int n,int n_in,const float* __restrict__  inputs,const float* __restrict__ weights,const float* __restrict__ bias,float* nodevals,float* preact){
  
    extern __shared__ float smem[];

    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + tid;

    for(int k=tid;k< n_in;k+=blockDim.x){
        smem[k]=inputs[k];
    }

    __syncthreads();
    if(i>= n)return;
    
    float val=bias[i];
    for(int j=0;j<n_in;j++){
        val+= weights[j*n+i]*smem[j];
    }

    preact[i]=val;
    nodevals[i]=activation(val);
     
}


__global__ void hiddenlayers(int n,int n_in,int w,int d,int p,int b,bool isout,const float* __restrict__ weights,const float* __restrict__ bias,float* preact,float* nodevals){
   extern __shared__ float smem[];
    int tid =threadIdx.x;
    int i =blockIdx.x*blockDim.x+tid;
    
    for(int k=tid ;k<n_in;k+=blockDim.x){
        smem[k]=nodevals[p+k];
    }
    __syncthreads();
    if(i>=n)return;

    float val=bias[b+i];
    for(int j=0;j<n_in;j++){
        val+=smem[j]*weights[w+i*n_in+j];
    }
    preact[d+i]=val;
    nodevals[d+i]=isout ? val:activation(val);
}


std::vector<float> forwardpass(std::vector<float> inputs){
   if(model.batching)printf("nomlp: use nomlp.train_batch(inputs) for batched training or disable batching in initlization \n ");return;
    if (host_layerdata.empty()) {
        std::fprintf(stderr, "cannot run prediction before initializing the network\n");
        return {};
    }
    if (inputs.size() != static_cast<size_t>(host_layerdata.front().node_in)) {
        std::fprintf(stderr, "input size does not match the network input size\n");
        return {};
    }
   cudaError_t err= cudaMemcpy(device_mlp_inputs,inputs.data(), inputs.size() * sizeof(float),cudaMemcpyHostToDevice);
   geterror("cudaMemcpy inputs to device", err);
   if (err != cudaSuccess) return {};
    int n =host_layerdata[0].node_out;
    int nin =host_layerdata[0].node_in;


     int threads = n<512?n :512;
    size_t smem_size=nin*sizeof(float);
    firstlayer<<<blocks(n,threads),threads,smem_size>>>(n,nin,device_mlp_inputs,device__weights,device__bias,device__nodevals,device__preact);
   

    for(int l=1;l<host_mlp_layer_count;l++){

        n=host_layerdata[l].node_out;
        nin=host_layerdata[l-1].node_out;
        int widx=host_layerdata[l].weight_index;
        int bidx =host_layerdata[l].bias_index;
        int didx=host_layerdata[l].nodedata_index;
        int pidx =host_layerdata[l-1].nodedata_index;
        bool isout= l== host_mlp_layer_count-1;

        threads=n<512?n:512;
        size_t smem= nin*sizeof(float);
        hiddenlayers<<<blocks(n,threads),threads,smem>>>(n,nin,widx,didx,pidx,bidx,isout,device__weights,device__bias,device__preact,device__nodevals);
       
    }
   n= host_layerdata[host_mlp_layer_count-1].nodedata_index;
   int count=host_layerdata[host_mlp_layer_count -1].node_out;

     err =cudaMemcpy(host_outputs.data(),device__nodevals+n,count*sizeof(float),cudaMemcpyDeviceToHost);
    geterror("cudaMemcpy outputs to host", err);
    if (err != cudaSuccess) return {};
   
    return host_outputs;
}


std::vector<float> nomlp::pridiction(std::vector<float>& inputs){
    return forwardpass(inputs);
}




