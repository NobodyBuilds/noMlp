#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "globals.h"
#include "adam.h"


__global__ void compute_delta(int n, int d, int l1_nout, int l1w, int l1_nin,
	int l1d, bool outlayer, const float *__restrict__ nodedata, float *dDelta,
	const float *__restrict__ dWeights, const float *__restrict__ dtarget,
	const float *__restrict__ dpreact, int nodedatasize)
{

	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int b = blockIdx.y;
	if (i >= n)
		return;
	if (outlayer)
	{
		const int outputOffset = b * nodedatasize + d;
		const int targetOffset = b * n;
		dDelta[outputOffset + i] = nodedata[outputOffset + i] - dtarget[targetOffset + i];
	}
	else
	{
		float sum = 0.0f;
		for (int k = 0; k < l1_nout; k++)
		{
			sum += dWeights[l1w + k * l1_nin + i] * dDelta[b * nodedatasize + l1d + k];
		}
		dDelta[b * nodedatasize + d + i] = sum * dslope(dpreact[b * nodedatasize + d + i]);
	}
}

__global__ void tuneweights(int n, int nin, int l, int l1d, int lw, int lsize,
	int d, int lb, const float *__restrict__ ddata,
	const float *__restrict__ dNodeData, float *dWeights, float *dDelta,
	float *dBias, int nodedatasize, int curbatch, float2 *adam_weights,
	float2 *adam_bias, float4 beta)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= n)
		return;

	float biasgrad = 0.0f;
	for (int b = 0; b < curbatch; b++)
		biasgrad += dDelta[b * nodedatasize + d + i];
	//dBias[lb + i] -= d_lr * (biasgrad / curbatch);
    adam_update(dBias[lb+i],adam_bias[lb+i],biasgrad/curbatch,d_lr,beta);
	for (int k = 0; k < nin; k++) 
	{
		float wgrad = 0.0f;
		for (int b = 0; b < curbatch; b++)
		{
			int off = b * nodedatasize;
			float nodeVal = (l == 0) ? ddata[b * nin + k] : dNodeData[off + l1d + k];
			wgrad += dDelta[off + d + i] * nodeVal;
		}
		const int weightIndex = (l == 0) ? lw + k * n + i : lw + i * lsize + k;
		adam_update(dWeights[weightIndex], adam_weights[weightIndex], wgrad / curbatch, d_lr, beta);
	}
}



void backpropogationbatch(int cb)
{
	int threads = 256;
	for (int l = host_mlp_layer_count - 1; l >= 0; l--)
	{

		int d = host_layerdata[l].nodedata_index;
		int n = host_layerdata[l].node_out;
		int l1out = 0, l1w = 0, l1d = 0, l1nin = 0;

		if (l < host_mlp_layer_count - 1)
		{
			l1out = host_layerdata[l + 1].node_out;
			l1w = host_layerdata[l + 1].weight_index;
			l1d = host_layerdata[l + 1].nodedata_index;
			l1nin = host_layerdata[l + 1].node_in;
		}

		bool outlayer = (l == host_mlp_layer_count - 1) ? 1 : 0;

		int blocks = (n + threads - 1) / threads;
		dim3 deltaGrid(blocks, cb);
		compute_delta<<<deltaGrid, threads>>>(n, d, l1out, l1w, l1nin, l1d, outlayer, device__nodevals, device__delta, device__weights, device_mlp_target, device__preact, host_nodedatabuffersize);
		
	}

	for (int l = host_mlp_layer_count - 1; l >= 0; l--)
	{
        float beta1 = 0.9f;
		float beta2 = 0.999f;
		float betat1 = 1.0f / (1.0f - powf(beta1, adam_step));
		float betat2 = 1.0f / (1.0f - powf(beta2, adam_step));
		int blocks = (host_layerdata[l].node_out + threads - 1) / threads;
		int nin = host_layerdata[l].node_in;
		int l1d = (l == 0) ? 0 : host_layerdata[l - 1].nodedata_index;
		int lw = host_layerdata[l].weight_index;
		int lsize = (l == 0) ? model.node_in : host_layerdata[l].node_in;
		int d = host_layerdata[l].nodedata_index;
		int lb = host_layerdata[l].bias_index;
		tuneweights<<<blocks, threads>>>(host_layerdata[l].node_out, nin, l, l1d, lw, lsize, d, lb, device_mlp_inputs, device__nodevals, device__weights, device__delta, device__bias, host_nodedatabuffersize, cb, adam_weights, adam_bias, {beta1, beta2, betat1, betat2});
        
		
	}
    adam_step++;
}
