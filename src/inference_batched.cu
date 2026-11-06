#include <cuda.h>
#include "noMlp.h"
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include "activations.h"
#include <algorithm>

/*
==BENCHMARK DETAILS==
-for inference_batch();

before optimizations:

	batchsize=64  :0.03sec/epoch
	batchsize=128 :0.02sec/epoch
	batchsize=256 :0.03sec/epoch
	batchsize=512 :0.027sec/epoch
	batchsize=1024:0.028sec/epoch

	dataset size =48,000 
	network 784->512->256->128->64->20


after optimizations:
*/

__global__ void firstlayerbatchinf(const int n, const int n_in, const int w, const float *__restrict__ input,
								   const float *__restrict__ dWeights, const float *__restrict__ dBias, float *dNodeData,
								   const int nodedatasize, int activation_type)
{
	extern __shared__ float smem[];

	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int b = blockIdx.y;
	if (i >= n)
		return;
	float val = dBias[i];
	for (int k = 0; k < n_in; k++)
	{
		float D = __ldg(&input[b * n_in + k]);
		val += dWeights[w + k * n + i] * D;
	}
	int off = b * nodedatasize;
	
	dNodeData[off + i] = activation(val, activation_type);
}

__global__ void solvelayersbatchinf(const int n, const int nin, const int w, const int d, const int b, const int p, bool isout, const float *__restrict__ dWeights, const float *__restrict__ dBias, float *dNodeData, const int nodedatasize, int activation_type)
{

	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int B = blockIdx.y;

	if (i >= n)
		return;

	float val = dBias[b + i];
	for (int j = 0; j < nin; j++)
	{
		float v = dNodeData[B * nodedatasize + p + j];
		val += v * dWeights[w + j * n + i];
	}
	int off = B * nodedatasize + d;
	
	dNodeData[off + i] = isout ? val : activation(val, activation_type);
}

bool runnetinf(int curBatch, noMlp &m)
{
	int n = m.layers[0].node_out;
	int threads = n > 512 ? 512 : n;

	int l1blocks = (n + threads - 1) / threads;
	int w = m.layers[0].weight_index;
	dim3 grid(l1blocks, curBatch);

	firstlayerbatchinf<<<grid, threads>>>(m.layers[0].node_out, m.layers[0].node_in, w, m.d_inputs, m.d_weights, m.d_bias, m.d_nodevals, m.nodedatabuffersize, m.activationInt);
	if (!geterror("launch first batch layer", cudaGetLastError()))
		return false;

	for (int l = 1; l < m.layers.size(); l++)
	{
		threads = m.layers[l].node_out > 512 ? 512 : m.layers[l].node_out;
		int blocks = (m.layers[l].node_out + threads - 1) / threads;
		int shared_bytes = m.layers[l].node_in * sizeof(float);

		int nin = m.layers[l - 1].node_out;
		int wl = m.layers[l].weight_index;
		int d = m.layers[l].nodedata_index;
		int b = m.layers[l].bias_index;
		int p = m.layers[l - 1].nodedata_index;

		bool isout = (l == m.layers.size() - 1);
		dim3 grid2(blocks, curBatch);
		solvelayersbatchinf<<<grid2, threads>>>(m.layers[l].node_out, nin, wl, d, b, p, isout, m.d_weights, m.d_bias, m.d_nodevals, m.nodedatabuffersize, m.activationInt);
		if (!geterror("launch batch layer", cudaGetLastError()))
			return false;
	}

	return true;
}

std::vector<float> noMlp::inference_batch(std::vector<float> &inputs)
{
	if(inputs.size() <= 0){
	
		std::printf("nomlp: input size shouldn't be zero or negative ,inputs size: (%d)\n", static_cast<int>(inputs.size()));
		return {};
	}

	if (!modeldata.batching)
	{
		std::printf("nomlp: enable batching in noMlp::init()\n");
		return {};
	}
	if (layers.empty() || modeldata.datasize <= 0 || modeldata.batchsize <= 0 ||
		d_inputs == nullptr || d_weights == nullptr ||
		d_bias == nullptr || d_nodevals == nullptr)
	{
		std::printf("nomlp: initialize the model and its buffers before inference_batch()\n");
		return {};
	}

	int datasize = static_cast<int>(inputs.size() / modeldata.node_in); // independent datasize ,allows any n inference samples to be processed, not just the original training dataset size
	int numbatches = (datasize + modeldata.batchsize - 1) / modeldata.batchsize;

	std::vector<float> batchInputs;

	try
	{
		batchInputs.resize(static_cast<size_t>(modeldata.batchsize) * modeldata.node_in);
	}
	catch (const std::exception &error)
	{
		std::printf("nomlp: could not allocate batch buffers: %s\n", error.what());
		return {};
	}
	for (int step = 0; step < numbatches; step++)
	{
		int curbatch = std::min(modeldata.batchsize, datasize - step * modeldata.batchsize);
		for (int b = 0; b < curbatch; ++b)
		{
			// no shuffling, just copy the inputs for the inference batch
			// keeping the original order of the inputs to match the expected output order
			const int sample = step * modeldata.batchsize + b;
			const size_t sourceOffset = static_cast<size_t>(sample) * modeldata.node_in;
			const size_t batchOffset = static_cast<size_t>(b) * modeldata.node_in;
			std::copy_n(inputs.data() + sourceOffset, modeldata.node_in, batchInputs.data() + batchOffset);
		}

		const size_t inputBytes = static_cast<size_t>(curbatch) * modeldata.node_in * sizeof(float);

		cudaError_t err = cudaMemcpy(d_inputs, batchInputs.data(), inputBytes, cudaMemcpyHostToDevice);
		if (!geterror("copy input batch to device", err))
			return {};

		if (!runnetinf(curbatch, *this))
			return {};

		// after each batch inference, copy the outputs back to host_outputs

		size_t hostoutputsize = static_cast<size_t>(inputs.size() / modeldata.node_in) * modeldata.node_out;
		host_outputs.resize(hostoutputsize); // resize host_outputs to hold all outputs for the entire dataset

		const size_t outputRowBytes =
			static_cast<size_t>(modeldata.node_out) * sizeof(float);

		err = cudaMemcpy2D(
			host_outputs.data() + static_cast<size_t>(step) * modeldata.batchsize * modeldata.node_out,
			outputRowBytes,
			d_nodevals + layers.back().nodedata_index,
			static_cast<size_t>(nodedatabuffersize) * sizeof(float),
			outputRowBytes,
			curbatch,
			cudaMemcpyDeviceToHost);
		if (!geterror("copy output batch to host", err))
			return {};
	}

	return this->host_outputs;
}