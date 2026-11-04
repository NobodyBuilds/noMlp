#include "noMlp.h"
#include "globals.h"
#include "activations.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "adam.h"

#include <cmath>


__global__ void compute_delta(
	int n, int data_offset, int next_layer_out, int next_layer_weight_offset,
	int next_layer_input_count, int next_layer_data_offset, bool output_layer,
	const float *node_values, float *deltas,
	const float *weights, const float *targets, const float *preactivations,
	int activation_type)
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
		sum += weights[ next_layer_weight_offset + i* next_layer_out + k] * deltas[next_layer_data_offset + k];
	}
	deltas[data_offset + i] = sum * dslope(preactivations[data_offset + i], activation_type);
}

__global__ void tuneweights(
	int output_count, int input_count, int layer_index, int previous_data_offset,
	int weight_offset, int weight_stride, int data_offset, int bias_offset,
	const float *inputs, const float *node_values, float *weights,
	const float *deltas, float *biases, float4 adam_beta, float2 *adam_weights, float2 *adam_bias,float d_lr)
{

	const int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= output_count)
		return;

	for (int k = 0; k < input_count; ++k)
	{
		const float node_data = layer_index == 0
									? inputs[k]
									: node_values[previous_data_offset + k];

		const int index =   weight_offset + k * output_count + i;

		// weights[index] -= d_lr * deltas[data_offset + i] * node_data;
		adam_update(weights[index], adam_weights[index], deltas[data_offset + i] * node_data, d_lr, adam_beta);
	}
	// biases[bias_offset + i] -= d_lr * deltas[data_offset + i];
	adam_update(biases[bias_offset + i], adam_bias[bias_offset + i], deltas[data_offset + i], d_lr, adam_beta);
}

static float backpropogation(noMlp& m, const std::vector<float> &targets)
{
    if (m.layers.empty() || m.d_target == nullptr || m.d_weights == nullptr ||
        m.d_bias == nullptr || m.d_nodevals == nullptr || m.d_delta == nullptr ||
        m.d_preact == nullptr || m.adam_weights == nullptr || m.adam_bias == nullptr)
	{
        std::printf("nomlp: initialize the model before running backpropagation\n");
		return 0.0f;
	}

    if (targets.empty() || targets.size() != static_cast<size_t>(m.modeldata.node_out))
	{
        std::printf("nomlp: target array size is empty or does not match the output size\n");
		return 0.0f;
	}
    for (float target : targets) {
        if (!std::isfinite(target)) {
            std::printf("nomlp: targets must contain finite values\n");
            return 0.0f;
        }
    }
	cudaError_t err = cudaMemcpy(
		m.d_target, targets.data(), targets.size() * sizeof(float),
		cudaMemcpyHostToDevice);
    if (!geterror("copy targets to device", err)) return 0.0f;

	constexpr int threads = 256;
	for (int l = m.layers_count - 1; l >= 0; --l)
	{
		const layerstruct &layer = m.layers[l];
		const bool output_layer = l == m.layers_count - 1;
		int next_layer_out = 0;
		int next_layer_weight_offset = 0;
		int next_layer_input_count = 0;
		int next_layer_data_offset = 0;

		if (!output_layer)
		{
			const layerstruct &next_layer = m.layers[l + 1];
			next_layer_out = next_layer.node_out;
			next_layer_weight_offset = next_layer.weight_index;
			next_layer_input_count = next_layer.node_in;
			next_layer_data_offset = next_layer.nodedata_index;
		}

		compute_delta<<<blocks(layer.node_out, threads), threads>>>(
			layer.node_out, layer.nodedata_index, next_layer_out,
			next_layer_weight_offset, next_layer_input_count,
			next_layer_data_offset, output_layer,
			m.d_nodevals, m.d_delta, m.d_weights,
			m.d_target, m.d_preact, m.activationInt);
		if (!geterror("launch delta kernel", cudaGetLastError())) return 0.0f;
	}

	for (int l = m.layers_count - 1; l >= 0; --l)
	{
		const layerstruct &layer = m.layers[l];
		const int previous_data_offset = l == 0
											 ? 0
											 : m.layers[l - 1].nodedata_index;

		float beta1 = 0.9f;
		float beta2 = 0.999f;
		float betat1 = 1.0f / (1.0f - powf(beta1, m.adam_step));
		float betat2 = 1.0f / (1.0f - powf(beta2, m.adam_step));
		tuneweights<<<blocks(layer.node_out, threads), threads>>>(
			layer.node_out, layer.node_in, l, previous_data_offset,
			layer.weight_index, layer.node_in, layer.nodedata_index,
			layer.bias_index, m.d_inputs, m.d_nodevals,
			m.d_weights, m.d_delta, m.d_bias, {beta1, beta2, betat1, betat2}, m.adam_weights, m.adam_bias,m.modeldata.lr);
		if (!geterror("launch weight-update kernel", cudaGetLastError())) return 0.0f;
	}
	m.adam_step++;
	if (!geterror("backpropagation synchronization", cudaDeviceSynchronize())) return 0.0f;

	float loss = 0.0f;

	for (int i = 0; i < m.modeldata.node_out; ++i)
	{
		const float difference = m.host_outputs[i] - targets[i];
		loss += 0.5f * difference * difference;
	}

	return loss;
}

float noMlp::backward(std::vector<float>& targets)
{

	return backpropogation(*this, targets);
}
