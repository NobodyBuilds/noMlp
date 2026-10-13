#include "noMlp.h"
#include "globals.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>

__device__ float activation(float x ){
    int type =noMlp.model.activation;
  
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

__global__ void firstlayer(const int n,int n_in,const float* __restrict__  inputs,const float* __restrict__ weights,const float* __restrict__ bias,float* nodevals,float* preact){
  
    //todo:: improve it add smem
    int i =blockIdx.x*blockDim.x+threadIdx.x;
    if(i>= n)return;

    float val=bias[i];
    for(int j=0;j<n_in;j++){
        val+= weights[j]*inputs[j];
    }

    preact[i]=val;
    nodevals[i]=activation(val);
     
}


__global__ void hiddenlayers(int n,int n_in,int w,int d,int p,int b,bool isout,const float* __restrict__ weights,const float* __restrict__ bias,float* preact,float* nodevals){

    int i =blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=n)return;

    float val=bias[b+i];
    for(int j=0;j<n_in;j++){
        val+=nodevals[p+j]*weights[w+j];
    }
    preact[d+i]=val;
    nodevals[d+i]=isout ? val:activation(val);
}


void forwardpass(){

    int n =host_layerdata[0].node_out;
    int nin =host_layerdata[0].node_in;
    firstlayer<<<blocks(n,n),n>>>(n,nin,device_mlp_inputs,device__weights,device__bias,device__nodevals,device__preact);


    for(l=1;l<host_mlp_layer_count;l++){

        int n=host_layerdata[l].node_out;
        int nin=host_layerdata[l-1].node_out;
        int widx=host_layerdata[l].weight_index;
        int bidx =host_layerdata[l].bias_index;
        int didx=host_layerdata[l].nodedata_index;
        int pidx =host_layerdata[l-1].nodedata_index;
        bool isout= l== host_mlp_layer_count-1;

        hiddenlayers<<<blocks(n,n),n>>>(n,nin,widx,didx,pidx,bidx,isout,device__weights,device__bias,device__preact,device__nodevals);
    }
}



__global__ void softmaxOutput(int n, float *dNodeData, int d, int nodedatasize)
{
    int b = blockIdx.x;
    extern __shared__ float sdata[];
    int tid = threadIdx.x;

    float val = (tid < n) ? dNodeData[b * nodedatasize + d + tid] : -1e30f;
    sdata[tid] = val;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] = fmaxf(sdata[tid], sdata[tid + s]);
        __syncthreads();
    }
    float maxVal = sdata[0];
    __syncthreads();

    float expVal = (tid < n) ? expf(val - maxVal) : 0.0f;
    sdata[tid] = expVal;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    float sumExp = sdata[0];
    __syncthreads();

    if (tid < n)
        dNodeData[b * nodedatasize + d + tid] = expVal / sumExp;
}


