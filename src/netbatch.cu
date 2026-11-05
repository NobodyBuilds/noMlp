#include "noMlp.h"
#include "globals.h"
#include "activations.h"
#include "backpropbatched.h"

#include <cuda_runtime.h>
#include <iostream>
#include <device_launch_parameters.h>
#include <algorithm>
#include <random>
#include <cmath>
#include <exception>

__global__ void getloss(int curBatch, const float *dNodeData, const float *dTarget,
						int d, int outputCount, int nodedatasize, float *loss)
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
			*loss += 0.5f * difference * difference;
		}
	}
}

__global__ void firstlayerbatch(const int n, const int n_in, const int w, const float *__restrict__ input,
								const float *__restrict__ dWeights, const float *__restrict__ dBias, float *preact, float *dNodeData,
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
	preact[off + i] = val;
	dNodeData[off + i] = activation(val, activation_type);
}

__global__ void solvelayersbatch(const int n, const int nin, const int w, const int d, const int b, const int p, bool isout, const float *__restrict__ dWeights, const float *__restrict__ dBias, float *preact, float *dNodeData, const int nodedatasize, int activation_type)
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
	preact[off + i] = val;
	dNodeData[off + i] = isout ? val : activation(val, activation_type);
}

bool runnet(int curBatch, noMlp &m)
{
	int n = m.layers[0].node_out;
	int threads = n > 512 ? 512 : n;

	int l1blocks = (n + threads - 1) / threads;
	int w = m.layers[0].weight_index;
	dim3 grid(l1blocks, curBatch);

	firstlayerbatch<<<grid, threads>>>(m.layers[0].node_out, m.layers[0].node_in, w, m.d_inputs, m.d_weights, m.d_bias, m.d_preact, m.d_nodevals, m.nodedatabuffersize, m.activationInt);
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
		solvelayersbatch<<<grid2, threads>>>(m.layers[l].node_out, nin, wl, d, b, p, isout, m.d_weights, m.d_bias, m.d_preact, m.d_nodevals, m.nodedatabuffersize, m.activationInt);
		if (!geterror("launch batch layer", cudaGetLastError()))
			return false;
	}

	const layerstruct &outputLayer = m.layers[m.layers.size() - 1];
	getloss<<<1, 1>>>(curBatch, m.d_nodevals, m.d_target, outputLayer.nodedata_index, outputLayer.node_out, m.nodedatabuffersize, m.loss);
	return geterror("launch batch loss", cudaGetLastError());
}

std::mt19937 shuffleRng(1234);
void shuffleindices(noMlp &m)
{
	m.indices.resize(m.modeldata.datasize);

	for (int i = 0; i < m.modeldata.datasize; i++)
	{
		m.indices[i] = i;
	}
	std::shuffle(m.indices.begin(), m.indices.end(), shuffleRng);
}
static bool first_itr = true;
float noMlp::train_batch(std::vector<float> &inputs, std::vector<float> &targets)
{

	if (!modeldata.batching)
	{
		std::printf("nomlp: enable batching in noMlp::init()\n");
		return 0.0f;
	}
	if (layers.empty() || modeldata.datasize <= 0 || modeldata.batchsize <= 0 ||
		d_inputs == nullptr || d_target == nullptr || d_weights == nullptr ||
		d_bias == nullptr || d_nodevals == nullptr || d_preact == nullptr ||
		d_delta == nullptr || loss == nullptr || adam_weights == nullptr || adam_bias == nullptr)
	{
		std::printf("nomlp: initialize the model and its buffers before train_batch()\n");
		return 0.0f;
	}
	const size_t expectedInputCount = static_cast<size_t>(modeldata.datasize) * modeldata.node_in;
	const size_t expectedTargetCount = static_cast<size_t>(modeldata.datasize) * modeldata.node_out;
	if (inputs.size() != expectedInputCount || targets.size() != expectedTargetCount)
	{
		std::printf("nomlp: train_batch expects datasize * node_in inputs and datasize * node_out targets\n");
		return 0.0f;
	}
	if (first_itr)
	{
		for (float input : inputs)
		{
			if (!std::isfinite(input))
			{
				std::printf("nomlp: inputs must contain finite values\n");
				return 0.0f;
			}
		}
		for (float target : targets)

		{
			if (!std::isfinite(target))
			{
				std::printf("nomlp: targets must contain finite values\n");
				return 0.0f;
			}
		}
		first_itr = false;//run data checking only once to avoid performance penalty on subsequent calls
	}
	int numbatches = (modeldata.datasize + modeldata.batchsize - 1) / modeldata.batchsize;
	try
	{
		shuffleindices(*this);
	}
	catch (const std::exception &error)
	{
		std::printf("nomlp: could not allocate batch indices: %s\n", error.what());
		return 0.0f;
	}
	if (!geterror("reset batch loss", cudaMemset(loss, 0, sizeof(float))))
		return 0.0f;
	float h_loss = 0.0f;
	std::vector<float> batchInputs;
	std::vector<float> batchTargets;
	try
	{
		batchInputs.resize(static_cast<size_t>(modeldata.batchsize) * modeldata.node_in);
		batchTargets.resize(static_cast<size_t>(modeldata.batchsize) * modeldata.node_out);
	}
	catch (const std::exception &error)
	{
		std::printf("nomlp: could not allocate batch buffers: %s\n", error.what());
		return 0.0f;
	}
	for (int step = 0; step < numbatches; step++)
	{
		int curbatch = std::min(modeldata.batchsize, modeldata.datasize - step * modeldata.batchsize);
		for (int b = 0; b < curbatch; ++b)
		{
			const int sample = indices[step * modeldata.batchsize + b];
			const size_t sourceOffset = static_cast<size_t>(sample) * modeldata.node_in;
			const size_t batchOffset = static_cast<size_t>(b) * modeldata.node_in;
			std::copy_n(inputs.data() + sourceOffset, modeldata.node_in, batchInputs.data() + batchOffset);
			const size_t targetSourceOffset = static_cast<size_t>(sample) * modeldata.node_out;
			const size_t targetBatchOffset = static_cast<size_t>(b) * modeldata.node_out;
			std::copy_n(targets.data() + targetSourceOffset, modeldata.node_out, batchTargets.data() + targetBatchOffset);
		}

		const size_t inputBytes = static_cast<size_t>(curbatch) * modeldata.node_in * sizeof(float);
		const size_t targetBytes = static_cast<size_t>(curbatch) * modeldata.node_out * sizeof(float);

		cudaError_t err = cudaMemcpy(d_inputs, batchInputs.data(), inputBytes, cudaMemcpyHostToDevice);
		if (!geterror("copy input batch to device", err))
			return 0.0f;

		err = cudaMemcpy(d_target, batchTargets.data(), targetBytes, cudaMemcpyHostToDevice);
		if (!geterror("copy target batch to device", err))
			return 0.0f;

		if (!runnet(curbatch, *this))
			return 0.0f;

		if (!backpropogationbatch(curbatch, *this))
			return 0.0f;
	}
	if (!geterror("batch training synchronization", cudaDeviceSynchronize()))
		return 0.0f;
	h_loss = *loss;

	return h_loss / modeldata.datasize;
}
