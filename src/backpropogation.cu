#include "globals.h"
#include "noMlp.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "adam.h"


__global__ void compute_delta(
	int n, int data_offset, int next_layer_out, int next_layer_weight_offset,
	int next_layer_input_count, int next_layer_data_offset, bool output_layer,
	const float *node_values, float *deltas,
	const float *weights, const float *targets, const float *preactivations)
{

	const int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= n)
		return;

	if (output_layer)
	{

		const float target = targets[i];
		deltas[data_offset + i] = node_values[data_offset + i] - target;

		return;
	}

	float sum = 0.0f;
	for (int k = 0; k < next_layer_out; ++k)
	{
		sum += weights[next_layer_weight_offset + k * next_layer_input_count + i] * deltas[next_layer_data_offset + k];
	}
	deltas[data_offset + i] = sum * dslope(preactivations[data_offset + i]);
}

__global__ void tuneweights(
	int output_count, int input_count, int layer_index, int previous_data_offset,
	int weight_offset, int weight_stride, int data_offset, int bias_offset,
	const float *inputs, const float *node_values, float *weights,
	const float *deltas, float *biases, float4 adam_beta, float2 *adam_weights, float2 *adam_bias)
{

	const int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= output_count)
		return;

	for (int k = 0; k < input_count; ++k)
	{
		const float node_data = layer_index == 0
									? inputs[k]
									: node_values[previous_data_offset + k];

		const int index = layer_index == 0
							  ? weight_offset + k * output_count + i
							  : weight_offset + i * weight_stride + k;
		// weights[index] -= d_lr * deltas[data_offset + i] * node_data;
		adam_update(weights[index], adam_weights[index], deltas[data_offset + i] * node_data, d_lr, adam_beta);
	}
	// biases[bias_offset + i] -= d_lr * deltas[data_offset + i];
	adam_update(biases[bias_offset + i], adam_bias[bias_offset + i], deltas[data_offset + i], d_lr, adam_beta);
}

static float backpropogation(const std::vector<float> &targets)
{
	if (host_layerdata.empty())
	{
		std::fprintf(stderr, "cannot run backpropagation before initializing the network\n");
		return 0.0f;
	}

	if (targets.size() == 0)
	{
		printf("target array size cannot be zero \n");
		return 0.0f;
	}
	cudaError_t err = cudaMemcpy(
		device_mlp_target, targets.data(), targets.size() * sizeof(float),
		cudaMemcpyHostToDevice);
	geterror("target memcpy to device", err);
	if (err != cudaSuccess)
	{

		return 0.0f;
	}

	constexpr int threads = 256;
	for (int l = host_mlp_layer_count - 1; l >= 0; --l)
	{
		const layerstruct &layer = host_layerdata[l];
		const bool output_layer = l == host_mlp_layer_count - 1;
		int next_layer_out = 0;
		int next_layer_weight_offset = 0;
		int next_layer_input_count = 0;
		int next_layer_data_offset = 0;

		if (!output_layer)
		{
			const layerstruct &next_layer = host_layerdata[l + 1];
			next_layer_out = next_layer.node_out;
			next_layer_weight_offset = next_layer.weight_index;
			next_layer_input_count = next_layer.node_in;
			next_layer_data_offset = next_layer.nodedata_index;
		}

		compute_delta<<<blocks(layer.node_out, threads), threads>>>(
			layer.node_out, layer.nodedata_index, next_layer_out,
			next_layer_weight_offset, next_layer_input_count,
			next_layer_data_offset, output_layer,
			device__nodevals, device__delta, device__weights,
			device_mlp_target, device__preact);
	}

	for (int l = host_mlp_layer_count - 1; l >= 0; --l)
	{
		const layerstruct &layer = host_layerdata[l];
		const int previous_data_offset = l == 0
											 ? 0
											 : host_layerdata[l - 1].nodedata_index;

		float beta1 = 0.9f;
		float beta2 = 0.999f;
		float betat1 = 1.0f / (1.0f - powf(beta1, adam_step));
		float betat2 = 1.0f / (1.0f - powf(beta2, adam_step));
		tuneweights<<<blocks(layer.node_out, threads), threads>>>(
			layer.node_out, layer.node_in, l, previous_data_offset,
			layer.weight_index, layer.node_in, layer.nodedata_index,
			layer.bias_index, device_mlp_inputs, device__nodevals,
			device__weights, device__delta, device__bias, {beta1, beta2, betat1, betat2}, adam_weights, adam_bias);
	}
	adam_step++;

	float loss = 0.0f;

	for (int i = 0; i < model.node_out; ++i)
	{
		const float difference = host_outputs[i] - targets[i];
		loss += 0.5f * difference * difference;
	}

	return loss;
}

float nomlp::backward(std::vector<float> targets)
{

	return backpropogation(targets);
}
