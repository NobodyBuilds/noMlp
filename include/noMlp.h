#pragma once

#include <vector>
#include <string>
#include <string_view>
#include <iostream>
#include <cstdio>

inline void geterror(const char* label, cudaError_t err) {
	if (err != cudaSuccess) {
		std::fprintf(stderr, " noMlp : %s failed: %s\n", label, cudaGetErrorString(err));
	}
}


struct modelst{
    int node_in;
    int node_out;
    float lr;
    int datasize;
    bool batching;
    int batchsize;
    std::string_view activation;
    const char* weights_adrs;
    const char* bias_adrs;
    std::vector<int> network;
};
   
   


class nomlp {
    private:


    public:
   
   
    void init(modelst model);
   
    std::vector<float> pridiction(std::vector<float>& inputs);
      float backward(std::vector<float> targets);

      void closeMlp();
};

inline nomlp noMlp;
