#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "noMlp.h"
#include "globals.h"
#include "activations.h"
#include "adam.h"
#include "backpropbatched.h"

#include <cmath>


__global__ void compute_delta(int n, int d, int l1_nout, int l1w, int l1_nin,
	int l1d, bool outlayer, const float *__restrict__ nodedata, float *dDelta,
	const float *__restrict__ dWeights, const float *__restrict__ dtarget,
	const float *__restrict__ dpreact, int nodedatasize, int activation_type)
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
			sum += dWeights[l1w + i * l1_nout + k] * dDelta[b * nodedatasize + l1d + k];
		}
		dDelta[b * nodedatasize + d + i] = sum * dslope(dpreact[b * nodedatasize + d + i], activation_type);
	}
}

__global__ void tuneweights(int n, int nin, int l, int l1d, int lw, int lsize,
	int d, int lb, const float *__restrict__ ddata,
	const float *__restrict__ dNodeData, float *dWeights, float *dDelta,
	float *dBias, int nodedatasize, int curbatch, float2 *adam_weights,
	float2 *adam_bias, float4 beta,float d_lr)
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
		const int weightIndex =lw + k * n + i ;
		adam_update(dWeights[weightIndex], adam_weights[weightIndex], wgrad / curbatch, d_lr, beta);
	}
}



bool backpropogationbatch(int cb, noMlp& m)
{
	if (cb <= 0 || m.layers.empty() || m.d_delta == nullptr ||
	    m.d_weights == nullptr || m.d_bias == nullptr || m.d_target == nullptr ||
	    m.d_inputs == nullptr || m.d_nodevals == nullptr || m.d_preact == nullptr ||
	    m.adam_weights == nullptr || m.adam_bias == nullptr) {
		std::printf("nomlp: invalid model state for batch backpropagation\n");
		return false;
	}

	int threads = 256;
	for (int l = m.layers_count - 1; l >= 0; l--)
	{

		int d = m.layers[l].nodedata_index;
		int n = m.layers[l].node_out;
		int l1out = 0, l1w = 0, l1d = 0, l1nin = 0;

		if (l < m.layers_count - 1)
		{
			l1out = m.layers[l + 1].node_out;
			l1w = m.layers[l + 1].weight_index;
			l1d = m.layers[l + 1].nodedata_index;
			l1nin = m.layers[l + 1].node_in;
		}

		bool outlayer = (l == m.layers_count - 1) ? 1 : 0;

		int blocks = (n + threads - 1) / threads;
		dim3 deltaGrid(blocks, cb);
		compute_delta<<<deltaGrid, threads>>>(n, d, l1out, l1w, l1nin, l1d, outlayer, m.d_nodevals, m.d_delta, m.d_weights, m.d_target, m.d_preact, m.nodedatabuffersize, m.activationInt);
		if (!geterror("launch batch delta kernel", cudaGetLastError())) return false;
	}

	for (int l = m.layers_count - 1; l >= 0; l--)
	{
		float betat1 = 1.0f / (1.0f - powf(beta1, m.adam_step));
		float betat2 = 1.0f / (1.0f - powf(beta2, m.adam_step));
		int blocks = (m.layers[l].node_out + threads - 1) / threads;
		int nin = m.layers[l].node_in;
		int l1d = (l == 0) ? 0 : m.layers[l - 1].nodedata_index;
		int lw = m.layers[l].weight_index;
		int lsize = (l == 0) ? m.modeldata.node_in : m.layers[l].node_in;
		int d = m.layers[l].nodedata_index;
		int lb = m.layers[l].bias_index;
		tuneweights<<<blocks, threads>>>(m.layers[l].node_out, nin, l, l1d, lw, lsize, d, lb, m.d_inputs, m.d_nodevals, m.d_weights, m.d_delta, m.d_bias, m.nodedatabuffersize, cb, m.adam_weights, m.adam_bias, {beta1, beta2, betat1, betat2}, m.modeldata.lr);
		if (!geterror("launch batch weight-update kernel", cudaGetLastError())) return false;
	}
    m.adam_step++;
	return true;
}
