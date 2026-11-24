#include "noMlp.h"
#include "globals.h"
#include "activations.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "adam.h"
#include "backpropkernels.cuh"
#include <cmath>
#include <vector>

std::vector<float> noMlp::backward(const std::vector<float> &targets, bool externeal_gradient)
{

	if (targets.size() != modeldata.batchsize * modeldata.node_out)
	{
		printf("invalid targets size \n");
		return {};
	}

	noMlp &m = *this;
	if (m.modeldata.batchsize <= 0 || m.layers.empty() || m.d_delta == nullptr ||
		m.d_weights == nullptr || m.d_bias == nullptr || m.d_target == nullptr ||
		m.d_inputs == nullptr || m.d_nodevals == nullptr || m.d_preact == nullptr ||
		m.adam_weights == nullptr || m.adam_bias == nullptr)
	{
		logerror("nomlp: invalid model state for  backward\n");
		return {};
	}
	size_t bytes = (modeldata.batchsize * modeldata.node_out) * sizeof(float);

	if (!geterror(" gradient copy to device ", cudaMemcpy(d_target, targets.data(), bytes, cudaMemcpyHostToDevice)))
	{
		return {};
	}

	int threads = 1024;

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

		if (!outlayer)
		{

			size_t layer_weight_size = l1nin * l1out;
			convertr2c<<<(layer_weight_size + 255) / 256, 256>>>(m.d_weights, m.column_major_weights, layer_weight_size, m.layers[l + 1].weight_index, m.layers[l + 1].node_out, l1nin);
		}
		threads = n > 1024 ? 1024 : n > 512 ? 512
											: 256;
		int blocks = (n + threads - 1) / threads;
		dim3 deltaGrid(blocks, m.modeldata.batchsize);
		compute_delta<<<deltaGrid, threads>>>(n, d, l1out, l1w, l1nin, l1d, outlayer, m.d_nodevals, m.d_delta, m.column_major_weights, m.d_target, m.d_preact, m.nodedatabuffersize, m.activationInt, externeal_gradient);
		if (!checkkernel("launch batch delta kernel"))
			return {};
	}

	host_weights_array.resize(modeldata.batchsize * nodedatabuffersize);

	if(!geterror("delta cpy to host", cudaMemcpy(&host_weights_array,d_delta,modeldata.batchsize * nodedatabuffersize*sizeof(float),cudaMemcpyDeviceToHost))){
		return {};
	}
	return host_weights_array;
}