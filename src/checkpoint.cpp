#include "checkpoint.h"
#include "adam.h"

#include <cuda_runtime.h>
#include <filesystem>
#include <fstream>
#include <cstdio>
#include <exception>
#include <sstream>
#include <string>
#include <vector>

namespace
{
template <typename T>
bool write_binary(const std::filesystem::path &path, const std::vector<T> &values)
{
    std::ofstream file(path, std::ios::binary | std::ios::trunc);
    if (!file)
        return false;

    file.write(reinterpret_cast<const char *>(values.data()),
               static_cast<std::streamsize>(values.size() * sizeof(T)));
    file.close();
    return static_cast<bool>(file);
}

template <typename T>
bool read_binary(const std::filesystem::path &path, std::size_t count,
                 std::vector<T> &values)
{
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file || static_cast<std::streamoff>(file.tellg()) !=
                     static_cast<std::streamoff>(count * sizeof(T)))
        return false;

    file.seekg(0);
    std::vector<T> loaded(count);
    file.read(reinterpret_cast<char *>(loaded.data()),
              static_cast<std::streamsize>(loaded.size() * sizeof(T)));
    if (!file)
        return false;

    values.swap(loaded);
    return true;
}

bool read_network(const std::filesystem::path &path, std::vector<int> &network)
{
    std::ifstream file(path);
    std::string line;
    while (std::getline(file, line))
    {
        constexpr char key[] = "Network Structure:";
        if (line.rfind(key, 0) != 0)
            continue;

        std::string value = line.substr(sizeof(key) - 1);
        for (char &ch : value)
            if (ch == '-' || ch == '>')
                ch = ' ';

        std::istringstream input(value);
        int nodes = 0;
        while (input >> nodes)
            network.push_back(nodes);
        return !network.empty();
    }
    return false;
}

bool read_adam_step(const std::filesystem::path &path, int &step)
{
    std::ifstream file(path);
    std::string line;
    while (std::getline(file, line))
    {
        constexpr char key[] = "Adam Step:";
        if (line.rfind(key, 0) == 0)
        {
            std::istringstream input(line.substr(sizeof(key) - 1));
            return (input >> step) && step > 0;
        }
    }
    return false;
}
} // namespace

void noMlp::save_checkpoint()
{
    try {
    if (modeldata.checkpoint_adrs == nullptr || *modeldata.checkpoint_adrs == '\0' ||
        d_weights == nullptr || d_bias == nullptr || adam_weights == nullptr || adam_bias == nullptr)
    {
        std::printf("nomlp: set a checkpoint address and initialize the model before saving\n");
        return;
    }

    const std::filesystem::path directory(modeldata.checkpoint_adrs);
    std::error_code error;
    std::filesystem::create_directories(directory, error);
    if (error)
    {
        std::printf("nomlp: could not create checkpoint directory: %s\n", error.message().c_str());
        return;
    }

    std::vector<float> weights(weightbuffersize), biases(biasbuffersize);
    std::vector<float2> adam_weight_state(weightbuffersize), adam_bias_state(biasbuffersize);
    cudaError_t result = cudaMemcpy(weights.data(), d_weights, weights.size() * sizeof(float), cudaMemcpyDeviceToHost);
    geterror("copy checkpoint weights", result);
    if (result != cudaSuccess)
        return;
    result = cudaMemcpy(biases.data(), d_bias, biases.size() * sizeof(float), cudaMemcpyDeviceToHost);
    geterror("copy checkpoint biases", result);
    if (result != cudaSuccess)
        return;
    result = cudaMemcpy(adam_weight_state.data(), adam_weights, adam_weight_state.size() * sizeof(float2), cudaMemcpyDeviceToHost);
    geterror("copy checkpoint Adam weights", result);
    if (result != cudaSuccess)
        return;
    result = cudaMemcpy(adam_bias_state.data(), adam_bias, adam_bias_state.size() * sizeof(float2), cudaMemcpyDeviceToHost);
    geterror("copy checkpoint Adam biases", result);
    if (result != cudaSuccess)
        return;

    if (!write_binary(directory / "weights.bin", weights) ||
        !write_binary(directory / "bias.bin", biases) ||
        !write_binary(directory / "adam_weights.bin", adam_weight_state) ||
        !write_binary(directory / "adam_bias.bin", adam_bias_state))
    {
        std::printf("nomlp: failed writing checkpoint data\n");
        return;
    }

    std::ofstream config(directory / "model_config.txt", std::ios::trunc);
    config << "Input Nodes: " << modeldata.node_in << '\n'
           << "Output Nodes: " << modeldata.node_out << '\n'
           << "Learning Rate: " << modeldata.lr << '\n'
           << "Data Size: " << modeldata.datasize << '\n'
           << "Batching: " << (modeldata.batching ? "Enabled" : "disabled") << '\n'
           << "Network Structure: ";
    for (std::size_t i = 0; i < modeldata.network.size(); ++i)
    {
        if (i != 0)
            config << " -> ";
        config << modeldata.network[i];
    }
    config << '\n'
           << "Batch Size: " << modeldata.batchsize << '\n'
           << "Activation Function: " << modeldata.activation << '\n';
    config.close();
    if (!config)
    {
        std::printf("nomlp: failed writing checkpoint config\n");
        return;
    }

    std::ofstream adam_config(directory / "adam_config.txt", std::ios::trunc);
    adam_config << "Beta 1: " << beta1 << '\n'
                << "Beta 2: " << beta2 << '\n'
                << "Adam Step: " << adam_step << '\n';
    adam_config.close();
    if (!adam_config)
    {
        std::printf("nomlp: failed writing Adam config\n");
        return;
    }

    host_weights_array.swap(weights);
    host_bias_array.swap(biases);
    std::printf("nomlp: checkpoint saved\n");
    } catch (const std::exception &error) {
        std::printf("nomlp: checkpoint save failed: %s\n", error.what());
    }
}

void noMlp::load_checkpoint()
{
    try {
    if (modeldata.checkpoint_adrs == nullptr || *modeldata.checkpoint_adrs == '\0' ||
        layers.empty() || d_weights == nullptr || d_bias == nullptr ||
        adam_weights == nullptr || adam_bias == nullptr)
    {
        std::printf("nomlp: set a checkpoint address and initialize the model before loading\n");
        return;
    }

    const std::filesystem::path directory(modeldata.checkpoint_adrs);
    std::vector<int> saved_network;
    if (!read_network(directory / "model_config.txt", saved_network) || saved_network != modeldata.network)
    {
        std::printf("nomlp: checkpoint network does not match the initialized model\n");
        return;
    }
    

    std::vector<float> weights, biases;
    std::vector<float2> adam_weight_state, adam_bias_state;
    int saved_adam_step = 0;
    if (!read_binary(directory / "weights.bin", weightbuffersize, weights) ||
        !read_binary(directory / "bias.bin", biasbuffersize, biases) ||
        !read_binary(directory / "adam_weights.bin", weightbuffersize, adam_weight_state) ||
        !read_binary(directory / "adam_bias.bin", biasbuffersize, adam_bias_state) ||
        !read_adam_step(directory / "adam_config.txt", saved_adam_step))
    {
        std::printf("nomlp: checkpoint files are missing or invalid\n");
        return;
    }

    cudaError_t result = cudaMemcpy(d_weights, weights.data(), weights.size() * sizeof(float), cudaMemcpyHostToDevice);
    geterror("load checkpoint weights", result);
    if (result != cudaSuccess)
        return;
    result = cudaMemcpy(d_bias, biases.data(), biases.size() * sizeof(float), cudaMemcpyHostToDevice);
    geterror("load checkpoint biases", result);
    if (result != cudaSuccess)
        return;
    result = cudaMemcpy(adam_weights, adam_weight_state.data(), adam_weight_state.size() * sizeof(float2), cudaMemcpyHostToDevice);
    geterror("load checkpoint Adam weights", result);
    if (result != cudaSuccess)
        return;
    result = cudaMemcpy(adam_bias, adam_bias_state.data(), adam_bias_state.size() * sizeof(float2), cudaMemcpyHostToDevice);
    geterror("load checkpoint Adam biases", result);
    if (result != cudaSuccess)
        return;

    adam_step = saved_adam_step;
    std::vector<float>().swap(host_weights_array);
    std::vector<float>().swap(host_bias_array);
    std::printf("nomlp: checkpoint loaded\n");
    } catch (const std::exception &error) {
        std::printf("nomlp: checkpoint load failed: %s\n", error.what());
    }
}
