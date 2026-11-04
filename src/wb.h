#pragma once
#include "noMlp.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <random>
#include <string>
#include <vector>
#include <filesystem>



inline void initWB(noMlp &m)
{

   
        m.host_weights_array.clear();
        m.host_bias_array.clear();
        m.host_weights_array.resize(m.weightbuffersize);
        m.host_bias_array.resize(m.biasbuffersize);

        std::mt19937 rng(42);
        std::normal_distribution<double> normal(0.0, 1.0);

        auto initializeNetwork = [&](
                                     std::vector<float> &weights,
                                     std::vector<float> &biases,
                                     const std::vector<layerstruct> &layers)
        {
            std::fill(biases.begin(), biases.end(), 0.0f);

            for (size_t l = 0; l < layers.size(); ++l)
            {
                const layerstruct &layer = layers[l];

                const int nin = layer.node_in;
                const int nout = layer.node_out;

                const int rows = std::max(nin, nout);
                const int cols = std::min(nin, nout);

                std::vector<double> q(rows * cols);

                for (double &value : q)
                    value = normal(rng);

                for (int c = 0; c < cols; ++c)
                {
                    for (int previous = 0; previous < c; ++previous)
                    {
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

                for (int i = 0; i < nout; ++i)
                {
                    for (int k = 0; k < nin; ++k)
                    {
                        const double value = nout >= nin
                                                 ? q[i * cols + k]
                                                 : q[k * cols + i];

                        const int index = layer.weight_index + k * nout + i;

                        weights[index] = gain * static_cast<float>(value);
                    }
                }
            }
        };

        initializeNetwork(
            m.host_weights_array, m.host_bias_array, m.layers);
    
    
}
