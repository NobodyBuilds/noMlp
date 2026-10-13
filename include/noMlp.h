#pragma once
#include <cuda.h>
#include <cuda_runtime.h>
#include "globals.h"
#include <vector>
#include <string>
#include <string_view>

class nomlp {
    private:


    public:
    modelst model;
    void init(float lr,int batchsize,std::string_view activation= "tanh");
    void initlayers(std::vector<int> neurons);
    void initlizeWeights();
    void loadWeights(const char* weights, const char* bias);

};

inline nomlp noMlp;