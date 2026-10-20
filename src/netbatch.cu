#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "globals.h"
#include "noMlp.h"
#include <algorithm>
#include <random>
#include <cmath>

__device__ float d_loss = 0.0f;
__global__ void getloss(int curBatch, const float *dNodeData, const float *dTarget,
	int d, int outputCount, int nodedatasize)
{
	if (blockIdx.x != 0 || threadIdx.x != 0)
		return;

	for (int b = 0; b < curBatch; ++b)
	{
		const int outputOffset = b * nodedatasize + d;
		const int targetOffset = b * outputCount;
		for (int i = 0; i < outputCount; ++i)
		{
			const float difference = dNodeData[outputOffset + i] - dTarget[targetOffset + i];
			d_loss += 0.5f * difference * difference;
		}
	}
}

__global__ void firstlayer(int n, int n_in, int w, const float *__restrict__ input,
						   const float *__restrict__ dWeights, const float *__restrict__ dBias, float *preact, float *dNodeData,
						   int nodedatasize)
{

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

void runnet(int curBatch)
{

	int threads = 256;

	int l1blocks = (host_layerdata[0].node_out + threads - 1) / threads;
	int w = host_layerdata[0].weight_index;
	dim3 grid(l1blocks, curBatch);

	firstlayer<<<grid, threads>>>(host_layerdata[0].node_out, host_layerdata[0].node_in, w, device_mlp_inputs, device__weights, device__bias, device__preact, device__nodevals, host_nodedatabuffersize);

	for (int l = 1; l < host_mlp_layer_count; l++)
	{

		int blocks = (host_layerdata[l].node_out + threads - 1) / threads;
		int shared_bytes = host_layerdata[l].node_in * sizeof(float);

		int nin = host_layerdata[l - 1].node_out;
		int wl = host_layerdata[l].weight_index;
		int d = host_layerdata[l].nodedata_index;
		int b = host_layerdata[l].bias_index;
		int p = host_layerdata[l - 1].nodedata_index;

		bool isout = (l == host_mlp_layer_count - 1);
		dim3 grid2(blocks, curBatch);
		solvelayers<<<grid2, threads>>>(host_layerdata[l].node_out, nin, wl, d, b, p, isout, device__weights, device__bias, device__preact, device__nodevals, host_nodedatabuffersize);
	}

	const layerstruct &outputLayer = host_layerdata[host_mlp_layer_count - 1];
	getloss<<<1, 1>>>(curBatch, device__nodevals, device_mlp_target, outputLayer.nodedata_index, outputLayer.node_out, host_nodedatabuffersize);
}

std::mt19937 shuffleRng(1234);
void shuffleindices()
{
	indices.resize(model.datasize);

	for (int i = 0; i < model.datasize; i++)
	{
		indices[i] = i;
	}
	std::shuffle(indices.begin(), indices.end(), shuffleRng);
}

float nomlp::train_batch(std::vector<float> &inputs, std::vector<float> &targets)
{

	if (!model.batching)
	{
		printf("nomlp: enable batching in nomlp.init() \n");
		return 0.0f;
	}
	if (host_layerdata.empty() || model.datasize <= 0 || model.batchsize <= 0)
	{
		printf("nomlp: initialize a non-empty dataset and network before train_batch() \n");
		return 0.0f;
	}
	const size_t expectedInputCount = static_cast<size_t>(model.datasize) * model.node_in;
	const size_t expectedTargetCount = static_cast<size_t>(model.datasize) * model.node_out;
	if (inputs.size() != expectedInputCount || targets.size() != expectedTargetCount)
	{
		printf("nomlp: train_batch expects datasize * node_in inputs and datasize * node_out targets \n");
		return 0.0f;
	}
	for (float target : targets)

	{
		if (!std::isfinite(target))
		{
			printf("nomlp: targets must contain finite values \n");
			return 0.0f;
		}
	}
	float zero=0.0f;
	cudaError_t resetErr = cudaMemcpyToSymbol(d_loss, &zero, sizeof(float));
	geterror("reset batch loss", resetErr);
	if (resetErr != cudaSuccess)
		return 0.0f;

	int numbatches = (model.datasize + model.batchsize - 1) / model.batchsize;
	shuffleindices();
	float h_loss = 0.0f;
	std::vector<float> batchInputs(static_cast<size_t>(model.batchsize) * model.node_in);
	std::vector<float> batchTargets(static_cast<size_t>(model.batchsize) * model.node_out);
	for (int step = 0; step < numbatches; step++)
	{
		int curbatch = std::min(model.batchsize, model.datasize - step * model.batchsize);
		for (int b = 0; b < curbatch; ++b)
		{
			const int sample = indices[step * model.batchsize + b];
			const size_t sourceOffset = static_cast<size_t>(sample) * model.node_in;
			const size_t batchOffset = static_cast<size_t>(b) * model.node_in;
			std::copy_n(inputs.data() + sourceOffset, model.node_in, batchInputs.data() + batchOffset);
			const size_t targetSourceOffset = static_cast<size_t>(sample) * model.node_out;
			const size_t targetBatchOffset = static_cast<size_t>(b) * model.node_out;
			std::copy_n(targets.data() + targetSourceOffset, model.node_out, batchTargets.data() + targetBatchOffset);
		}

		const size_t inputBytes = static_cast<size_t>(curbatch) * model.node_in * sizeof(float);
		const size_t targetBytes = static_cast<size_t>(curbatch) * model.node_out * sizeof(float);

		cudaError_t err = cudaMemcpy(device_mlp_inputs, batchInputs.data(), inputBytes, cudaMemcpyHostToDevice);
		geterror("input batch memcpy to device", err);
		if (err != cudaSuccess)
			return 0.0f;

		err = cudaMemcpy(device_mlp_target, batchTargets.data(), targetBytes, cudaMemcpyHostToDevice);
		geterror("target batch memcpy to device", err);
		if (err != cudaSuccess)
			return 0.0f;

		runnet(curbatch);

		backpropogationbatch(curbatch);

	}
	cudaError_t err = cudaMemcpyFromSymbol(&h_loss, d_loss, sizeof(float));
	geterror("loss memcpy to host", err);
	if (err != cudaSuccess)
		return 0.0f;

	return h_loss / model.datasize;
}
