#include "noMlp.h"
#include "globals.h"
#include "activations.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>
#include <random>
#include <algorithm>
#include <cmath>


__global__ void firstlayer(const int n,int n_in,const float* __restrict__  inputs,const float* __restrict__ weights,const float* __restrict__ bias,float* nodevals,float* preact,int activation_type){
  
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
    nodevals[i]=activation(val,activation_type);
     
}


__global__ void hiddenlayers(int n,int n_in,int w,int d,int p,int b,bool isout,const float* __restrict__ weights,const float* __restrict__ bias,float* preact,float* nodevals,int activation_type){
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
        val+=smem[j]*weights[w+j*n+i];
    }
    preact[d+i]=val;
    nodevals[d+i]=isout ? val:activation(val,activation_type);
}



std::vector<float> forwardpass(noMlp& m, const std::vector<float> &inputs){
   if(m.modeldata.batching) {
       std::printf("nomlp: use train_batch() for batched training or disable batching in initialization\n");
       return {};
   }
    if (m.layers.empty() || m.d_inputs == nullptr || m.d_weights == nullptr ||
        m.d_bias == nullptr || m.d_nodevals == nullptr || m.d_preact == nullptr) {
        std::printf("nomlp: initialize the model before running prediction\n");
        return {};
    }
    if (inputs.size() != static_cast<size_t>(m.layers.front().node_in)) {
        std::printf("nomlp: input size does not match the network input size\n");
        return {};
    }
    for (float input : inputs) {
        if (!std::isfinite(input)) {
            std::printf("nomlp: inputs must contain finite values\n");
            return {};
        }
    }
   cudaError_t err= cudaMemcpy(m.d_inputs,inputs.data(), inputs.size() * sizeof(float),cudaMemcpyHostToDevice);
   if (!geterror("copy inputs to device", err)) return {};
    int n =m.layers[0].node_out;
    int nin =m.layers[0].node_in;


     int threads = n<512?n :512;
    size_t smem_size=nin*sizeof(float);
    firstlayer<<<blocks(n,threads),threads,smem_size>>>(n,nin,m.d_inputs,m.d_weights,m.d_bias,m.d_nodevals,m.d_preact,m.activationInt);
    if (!geterror("launch first-layer kernel", cudaGetLastError())) return {};
   

    for(int l=1;l<m.layers_count;l++){

        n=m.layers[l].node_out;
        nin=m.layers[l-1].node_out;
        int widx=m.layers[l].weight_index;
        int bidx =m.layers[l].bias_index;
        int didx=m.layers[l].nodedata_index;
        int pidx =m.layers[l-1].nodedata_index;
        bool isout= l== m.layers_count-1;

        threads=n<512?n:512;
        size_t smem= nin*sizeof(float);
        hiddenlayers<<<blocks(n,threads),threads,smem>>>(n,nin,widx,didx,pidx,bidx,isout,m.d_weights,m.d_bias,m.d_preact,m.d_nodevals,m.activationInt);
        if (!geterror("launch hidden-layer kernel", cudaGetLastError())) return {};
       
    }
   n= m.layers[m.layers_count-1].nodedata_index;
   int count=m.layers[m.layers_count -1].node_out;



     err =cudaMemcpy(m.host_outputs.data(),m.d_nodevals+n,count*sizeof(float),cudaMemcpyDeviceToHost);
    if (!geterror("copy outputs to host", err)) return {};
   
    return m.host_outputs;
}


std::vector<float> noMlp::pridiction(std::vector<float>& inputs){
    return forwardpass(*this,inputs);
}




