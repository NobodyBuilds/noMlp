#include "noMlp.h"
#include "globals.h"
#include "activations.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>
#include <random>
#include <algorithm>
#include <cmath>
#include "softmax.cuh"
#include "train_batch.cuh"


std::vector<float> noMlp::forward(const std::vector<float>&inputs){

    noMlp & m=*this;
    if(inputs.size() != m.modeldata.node_in* m.modeldata.batchsize) {
        printf("the inputs size does not match to batchsize * inputs \n");
         return{};
    }
    if(!geterror(" inputs memcpy ",cudaMemcpy(m.d_inputs,inputs.data(),inputs.size()*sizeof(float),cudaMemcpyHostToDevice))){
        return {};
    }

   int n= m.layers[0].node_out;
    int threads= n>1024?1024: n>512? 512: 256;
   int dim= (n+threads-1)/threads;
   dim3 grid(dim,m.modeldata.batchsize);
   int w= m.layers[0].weight_index;
    firstlayerbatch<<<grid,threads>>>(n,m.layers[0].node_in,w, m.d_inputs,m.d_weights,m.d_bias,m.d_preact,m.d_nodevals,m.nodedatabuffersize,m.activationInt);

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
		dim3 grid2(blocks, m.modeldata.batchsize);
		solvelayersbatch<<<grid2, threads>>>(m.layers[l].node_out, nin, wl, d, b, p, isout, m.d_weights, m.d_bias, m.d_preact, m.d_nodevals, m.nodedatabuffersize, m.activationInt);
		
	}
    
    int d=m.layers.back().nodedata_index;
    if(!geterror("mem cpy back to host",cudaMemcpy2D(&m.host_outputs,
        m.modeldata.node_out *sizeof(float),
        m.d_nodevals+d,
        static_cast<size_t>(nodedatabuffersize) * sizeof(float) ,
        m.modeldata.node_out*sizeof(float),
        m.modeldata.batchsize,
    cudaMemcpyDeviceToHost))){
        return{};
    }
    return m.host_outputs;

}
