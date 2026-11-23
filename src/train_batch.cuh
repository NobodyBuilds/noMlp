#pragma once 
#include <cuda_runtime.h>


__global__ void firstlayerbatch(const int n, const int n_in, const int w, const float *__restrict__ input,
								const float *__restrict__ dWeights, const float *__restrict__ dBias, float *preact, float *dNodeData,
								const int nodedatasize, int activation_type);


 __global__ void solvelayersbatch(const int n, const int nin, const int w, const int d, const int b, const int p, bool isout, const float *__restrict__ dWeights, const float *__restrict__ dBias, float *preact, float *dNodeData, const int nodedatasize, int activation_type);
                               