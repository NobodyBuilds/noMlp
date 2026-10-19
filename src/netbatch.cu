#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "globals.h"
#include "noMlp.h"
#include <algorithm>
#include <random>

__global__ void firstlayer(int n, int w, int step, int *batchidx, const float *__restrict__ input, 
	const float *__restrict__ dWeights, const float *__restrict__ dBias, float *preact, float *dNodeData,
	 int nodedatasize)
{


	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int b = blockIdx.y;
	if (i >= n)
		return;
	float val = dBias[i];
	for (int k = 0; k < dmodel.inputnode; k++)
	{
		float D = __ldg(&input[batchidx[step * dModel.batchsize + b] * dModel.inputnode + k]);
		val += dWeights[w + i * dModel.inputnode + k] * D;
	}
	int off = b * nodedatasize;
	preact[off + i] = val;
	dNodeData[off + i] = activation(val);
}

__global__ void solvelayers(int n, int nin, int w, int d, int b, int p, bool isout, const float *__restrict__ dWeights, const float *__restrict__ dBias, float *preact, float *dNodeData, int nodedatasize)
{

	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int B = blockIdx.y;

	if (i >= n)
		return;

	float val = dBias[b + i];
	for (int j = 0; j < nin; j++)
	{
		float v = dNodeData[B * nodedatasize + p + j];
		val += v * dWeights[w + i * nin + j];
	}
	int off = B * nodedatasize + d;
	preact[off + i] = val;
	dNodeData[off + i] = isout ? val : activation(val);
}

void runnet(int curBatch, int step )
{

	int threads = 256;

	int l1blocks = (host_layerdata[0].node_out + threads - 1) / threads;
	int w = host_layerdata[0].weight_index;
	dim3 grid(l1blocks, curBatch);
	
		firstlayer<<<grid, threads>>>(host_layerdata[0].node_out, w, step, batchidx, dData, dWeights, dBias, dpreact, dNodeData, nodeDataSize);
	

	for (int l = 1; l < layers; l++)
	{

		int blocks = (layer[l].Nout + threads - 1) / threads;
		int shared_bytes = layer[l].Nin * sizeof(float);

		int nin = layer[l - 1].Nout;
		int wl = layer[l].wIdx;
		int d = layer[l].dIdx;
		int b = layer[l].bIdx;
		int p = layer[l - 1].dIdx;

		bool isout = (l == layers - 1);
		dim3 grid2(blocks, curBatch);
		solvelayers<<<grid2, threads>>>(layer[l].Nout, nin, wl, d, b, p, isout, dWeights, dBias, dpreact, dNodeData, nodeDataSize);

		
	}


	
		getloss<<<1, 1>>>(curBatch, step, valbatchidx, dNodeData, d_valTarget, layer[layers - 1].dIdx, nodeDataSize, false);
	
}

std::mt19937 shuffleRng(1234);
void shuffleindices()
{
	indices.resize(model.datasize);
	
	for (int i = 0; i < model.datasize; i++){
		indices[i] = i;
    }
    std::shuffle(indices.begin(), indices.end(), shuffleRng);
	geterror("indices memcpy to device ",cudaMemcpy(batchidx,indices.data(),model.batchsize*sizeof(int),cudaMemcpyHostToDevice));	
}

float train_batch(std::vector<float>&inputs,std::vector<float>& targets){
    if(!model.batching)printf("nomlp: enable batching in nomlp.init() \n"); return;
    shuffleindices();
    int numbatches =(model.datasize + model.batchsize-1)/model.batchsize;
    

int offset=0;

    for(int step =0;step<numbatches ;step++){
        int size=(model.datasize/model.batchsize)*model.node_in ;
         offset = step==0? 0: size*step;
        geterror("input batch memcpy to device ",cudaMemcpy(device_mlp_inputs,inputs.data()+offset,size*sizeof(float),cudaMemcpyHostToDevice));


    }

}