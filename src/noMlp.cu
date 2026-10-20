#include "noMlp.h"
#include "globals.h"
#include <vector>
#include <iostream>
#include <random>
#include <fstream>
#include <string>
#include <cstdio>
#include <string_view>
#include <algorithm>
#include <cmath>
#include "adam.h"
void addlayer(int node_in,int node_out){
    layerstruct l;
    l={0};
    l.node_in=node_in;
    l.node_out=node_out;
    host_layerdata.push_back(l);
    host_mlp_layer_count++;
    host_weightbuffersize+= node_in*node_out;
    host_nodedatabuffersize+=node_out;
    host_biasbuffersize+=node_out;
}
void setoffsets(){
    int weight_offset = 0;
    int node_offset = 0;
    int bias_offset = 0;

    for (layerstruct& layer : host_layerdata) {
        layer.weight_index = weight_offset;
        layer.nodedata_index = node_offset;
        layer.bias_index = bias_offset;

        weight_offset += layer.node_in * layer.node_out;
        node_offset += layer.node_out;
        bias_offset += layer.node_out;
    }
}

void readfiles(const char* path, std::vector<float>& container)
{
	container.clear();
	if (path == nullptr || *path == '\0') {
		return;
	}

	std::fstream file(path, std::ios::in);
	if (!file.is_open())
	{
		std::cerr << "Error opening " << path << " file" << std::endl;
		return;
	}

	std::string val;

	while (file >> val)
	{
		container.push_back(std::stof(val));

	}
}

void initWB(){
   bool weightsloaded = true;
    readfiles(model.weights_adrs,host_weights_array);
    if(host_weights_array.size()!= host_weightbuffersize){
       weightsloaded=false;
       printf("noMlp: weights size mismatch \n");
    }
    readfiles(model.bias_adrs,host_bias_array);
    if(host_bias_array.size()!= host_biasbuffersize){
        weightsloaded=false;
        printf("noMlp: bias size mismatch \n");
    }

if (!weightsloaded) {
	printf("noMlp: weights loading error ,generating new weights\n");
	
	host_weights_array.clear();
	host_bias_array.clear();
	host_weights_array.resize(host_weightbuffersize);
	host_bias_array.resize(host_biasbuffersize);

	
	std::mt19937 rng(42);
	std::normal_distribution<double> normal(0.0, 1.0);

	auto initializeNetwork = [&](
		std::vector<float>& weights,
		std::vector<float>& biases,
		const std::vector<layerstruct>& layers
		)
		{
			std::fill(biases.begin(), biases.end(), 0.0f);

			for (size_t l = 0; l < layers.size(); ++l) {
				const layerstruct& layer = layers[l];

				const int nin = layer.node_in;
				const int nout = layer.node_out;

				const int rows = std::max(nin, nout);
				const int cols = std::min(nin, nout);

				std::vector<double> q(rows * cols);

				for (double& value : q)
					value = normal(rng);

				for (int c = 0; c < cols; ++c) {
					for (int previous = 0; previous < c; ++previous) {
						double projection = 0.0;

						for (int r = 0; r < rows; ++r)
							projection +=
							q[r * cols + c] *
							q[r * cols + previous];

						for (int r = 0; r < rows; ++r)
							q[r * cols + c] -=
							projection * q[r * cols + previous];
					}

					double normSquared = 0.0;

					for (int r = 0; r < rows; ++r)
						normSquared +=
						q[r * cols + c] * q[r * cols + c];

					const double inverseNorm =
						1.0 / std::sqrt(std::max(normSquared, 1e-24));

					for (int r = 0; r < rows; ++r)
						q[r * cols + c] *= inverseNorm;
				}

				const bool finalLayer = l + 1 == layers.size();

				const float gain = finalLayer
					? 1.0f
					: std::sqrt(2.0f);

				for (int i = 0; i < nout; ++i) {
					for (int k = 0; k < nin; ++k) {
						const double value = nout >= nin
							? q[i * cols + k]
							: q[k * cols + i];

						const int index = l == 0
							? layer.weight_index + k * nout + i
							: layer.weight_index + i * nin + k;

						weights[index] = gain * static_cast<float>(value);
					}
				}
			}
		};

	initializeNetwork(
		host_weights_array, host_bias_array, host_layerdata);

	

}
else {
	printf("noMlp: weights loaded from files\n");
}

}

void allocatemlpmemory(){
    geterror("cudaMalloc device weights", cudaMalloc(&device__weights,host_weightbuffersize *sizeof(float)));
    geterror("cudaMalloc device bias", cudaMalloc(&device__bias,host_biasbuffersize *sizeof(float)));
    geterror("cudaMalloc device node values", cudaMalloc(&device__nodevals,model.batchsize* host_nodedatabuffersize *sizeof(float)));
    geterror("cudaMalloc device preactivations", cudaMalloc(&device__preact,model.batchsize* host_nodedatabuffersize *sizeof(float)));
    geterror("cudaMalloc device deltas", cudaMalloc(&device__delta,model.batchsize* host_nodedatabuffersize *sizeof(float)));
    const int input_capacity = model.batchsize;
    const int target_capacity = model.batching ? model.batchsize * model.node_out : model.node_out;
    geterror("cudaMalloc device inputs", cudaMalloc(&device_mlp_inputs, static_cast<size_t>(input_capacity) * model.node_in * sizeof(float)));
    geterror("cudaMalloc device targets", cudaMalloc(&device_mlp_target, static_cast<size_t>(target_capacity) * sizeof(float)));

    geterror("cudaMemcpy weights to device", cudaMemcpy(device__weights,host_weights_array.data(),host_weightbuffersize*sizeof(float),cudaMemcpyHostToDevice));
    geterror("cudaMemcpy bias to device", cudaMemcpy(device__bias,host_bias_array.data(),host_biasbuffersize*sizeof(float),cudaMemcpyHostToDevice));

}



bool initlayers(const std::vector<int>& neurons){
    if (neurons.size() < 2) {
        std::fprintf(stderr, "network must contain at least input and output sizes\n");
        return false;
    }

    for (int node_count : neurons) {
        if (node_count <= 0) {
            std::fprintf(stderr, "network layer sizes must be positive\n");
            return false;
        }
    }

    host_layerdata.clear();
    host_mlp_layer_count = 0;
    host_weightbuffersize = 0;
    host_nodedatabuffersize = 0;
    host_biasbuffersize = 0;
   
    for (size_t i = 0; i + 1 < neurons.size(); ++i) {
        addlayer(neurons[i], neurons[i + 1]);
    }

    setoffsets();
    host_outputs.resize(neurons.back());
    return true;
}


void setmodel(const modelst& a) {
    model = a;

    if (model.activation == "tanh")
        activationInt = 1;
    else if (model.activation == "relu")
        activationInt = 2;
    else if (model.activation == "leakyrelu")
        activationInt = 3;
    else
        activationInt = 1; // Default: tanh

    model.batchsize = model.batching ? model.batchsize : 1;

    geterror("cudaMemcpyToSymbol lr to device", cudaMemcpyToSymbol(d_lr,&model.lr,sizeof(float)));
    geterror("cudaMemcpyToSymbol activation", cudaMemcpyToSymbol(dactivationint,&activationInt,sizeof(activationInt)));

}

void nomlp::init(modelst Model ){
    if (Model.network.size() >= 2 &&
        (Model.node_in != Model.network.front() || Model.node_out != Model.network.back())) {
        std::fprintf(stderr, "model input/output sizes must match the network endpoints\n");
        return;
    }
    if (Model.batching && Model.batchsize <= 0) {
        std::fprintf(stderr, "batchsize must be positive\n");
        return;
    }
    if (!initlayers(Model.network)) {
        return;
    }

    setmodel(Model);
    initWB();
    allocatemlpmemory();
    initadam(host_weightbuffersize,host_biasbuffersize);
    
}


void nomlp::closeMlp(){
    if (device_mlp_inputs != nullptr) {
        const cudaError_t err = cudaFree(device_mlp_inputs);
        geterror("cudaFree device inputs", err);
        if (err == cudaSuccess) device_mlp_inputs = nullptr;
    }
    if (device_mlp_target != nullptr) {
        const cudaError_t err = cudaFree(device_mlp_target);
        geterror("cudaFree device targets", err);
        if (err == cudaSuccess) device_mlp_target = nullptr;
    }
    if (device_indices != nullptr) {
        const cudaError_t err = cudaFree(device_indices);
        geterror("cudaFree device indices", err);
        if (err == cudaSuccess) device_indices = nullptr;
    }
    if (device__weights != nullptr) {
        const cudaError_t err = cudaFree(device__weights);
        geterror("cudaFree device weights", err);
        if (err == cudaSuccess) device__weights = nullptr;
    }
    if (device__bias != nullptr) {
        const cudaError_t err = cudaFree(device__bias);
        geterror("cudaFree device bias", err);
        if (err == cudaSuccess) device__bias = nullptr;
    }
    if (device__nodevals != nullptr) {
        const cudaError_t err = cudaFree(device__nodevals);
        geterror("cudaFree device node values", err);
        if (err == cudaSuccess) device__nodevals = nullptr;
    }
    if (device__preact != nullptr) {
        const cudaError_t err = cudaFree(device__preact);
        geterror("cudaFree device preactivations", err);
        if (err == cudaSuccess) device__preact = nullptr;
    }
    if (device__delta != nullptr) {
        const cudaError_t err = cudaFree(device__delta);
        geterror("cudaFree device deltas", err);
        if (err == cudaSuccess) device__delta = nullptr;
    }

    clearadam();
}




