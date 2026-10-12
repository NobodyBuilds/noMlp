#pragma once
#include <cuda.h>
#include <cuda_runtime.h>
#include "globals.h"
#include <vector>

class nomlp {
    private:


    public:

    void initlayers(std::vector<int> neurons);

};

inline nomlp noMlp;