#include "noMlp.h"
#include "globals.h"
#include "layers.h"
#include "wb.h"
#include <cuda_runtime.h>
#include <vector>
#include <iostream>
#include <random>
#include <fstream>
#include <string>
#include <cstdio>
#include <string_view>
#include <algorithm>
#include <cmath>
#include <exception>
#include <utility>
#include "adam.h"
#include "memory.h"








bool setmodelactivation(noMlp& m) {
    modelconfig& model = m.modeldata;

    if (model.activation == "tanh")
        m.activationInt = 1;
    else if (model.activation == "relu")
        m.activationInt = 2;
    else if (model.activation == "leakyrelu")
        m.activationInt = 3;
    else
    {
        std::printf("nomlp: unsupported activation; use tanh, relu, or leakyrelu\n");
        return false;
    }

    model.batchsize = model.batching ? model.batchsize : 1;
    return true;
}

void noMlp::init(modelconfig& model ){
    modelconfig config;
    try {
        config = model;
    } catch (const std::exception &error) {
        std::printf("nomlp: could not copy model configuration: %s\n", error.what());
        return;
    }
    close();
    if (d_inputs != nullptr || d_target != nullptr || d_weights != nullptr ||
        d_bias != nullptr || d_nodevals != nullptr || d_preact != nullptr ||
        d_delta != nullptr || loss != nullptr || adam_weights != nullptr ||
        adam_bias != nullptr) {
        std::printf("nomlp: could not release existing buffers; initialization stopped\n");
        return;
    }
    modeldata = std::move(config);
    if (modeldata.network.size() >= 2 &&
        (modeldata.node_in !=modeldata.network.front() || modeldata.node_out != modeldata.network.back())) {
        std::printf("nomlp: model input/output sizes must match the network endpoints\n");
        return;
    }
    if (modeldata.batching && modeldata.batchsize <= 0) {
        std::printf("nomlp: batch size must be positive\n");
        return;
    }
    try {
        if (!initlayers(modeldata.network, *this)) return;
        host_outputs.resize(modeldata.node_out);
        if (!setmodelactivation(*this)) {
            close();
            return;
        }
        initWB(*this);
    } catch (const std::exception &error) {
        std::printf("nomlp: model initialization failed: %s\n", error.what());
        close();
        return;
    }
    if (!allocatemlpmemory(*this)) {
        close();
        return;
    }
    if (!initadam(*this)) {
        close();
        return;
    }
    
}





void noMlp::close(){




    if (d_inputs != nullptr) {
        const cudaError_t err = cudaFree(d_inputs);
        geterror("cudaFree device inputs", err);
        if (err == cudaSuccess) d_inputs = nullptr;
    }
    if (d_target != nullptr) {
        const cudaError_t err = cudaFree(d_target);
        geterror("cudaFree device targets", err);
        if (err == cudaSuccess) d_target = nullptr;
    }

    if (d_weights != nullptr) {
        const cudaError_t err = cudaFree(d_weights);
        geterror("cudaFree device weights", err);
        if (err == cudaSuccess) d_weights = nullptr;
    }
    if (d_bias != nullptr) {
        const cudaError_t err = cudaFree(d_bias);
        geterror("cudaFree device bias", err);
        if (err == cudaSuccess) d_bias = nullptr;
    }
    if (d_nodevals != nullptr) {
        const cudaError_t err = cudaFree(d_nodevals);
        geterror("cudaFree device node values", err);
        if (err == cudaSuccess) d_nodevals = nullptr;
    }
    if (d_preact != nullptr) {
        const cudaError_t err = cudaFree(d_preact);
        geterror("cudaFree device preactivations", err);
        if (err == cudaSuccess) d_preact = nullptr;
    }
    if (d_delta != nullptr) {
        const cudaError_t err = cudaFree(d_delta);
        geterror("cudaFree device deltas", err);
        if (err == cudaSuccess) d_delta = nullptr;
    }
    if (loss != nullptr) {
        const cudaError_t err = cudaFree(loss);
        geterror("cudaFree loss", err);
        if (err == cudaSuccess) loss = nullptr;
    }
    clearadam(*this);

    cudaFree(column_major_weights);

    std::vector<layerstruct>().swap(layers);
    std::vector<float>().swap(host_outputs);
    std::vector<float>().swap(host_weights_array);
    std::vector<float>().swap(host_bias_array);
    std::vector<int>().swap(indices);
    layers_count = 0;
    weightbuffersize = 0;
    biasbuffersize = 0;
    nodedatabuffersize = 0;
    modeldata = modelconfig{};
}




