# noMlp

noMlp is a small multilayer perceptron (MLP) library for C++ and CUDA. It supports single-sample prediction and batched training, with weights and optimizer state stored per model instance.

## Requirements

- Windows
- NVIDIA CUDA Toolkit with `nvcc`
- Visual Studio 2022 C++ x64 build tools
- C++17
- An NVIDIA GPU supported by the installed CUDA Toolkit

## Build the library

From the repository root, run:

```powershell
.\compile\compile.bat
```

The script compiles the `.cu` and `.cpp` files in `src` and writes the static library to `lib\noMlp.lib`. It currently looks for the Visual Studio Community `vcvars64.bat` at the default 2022 installation path. If your Visual Studio installation uses a different edition or location, update `VCVARS` in `compile\compile.bat`.

For a CUDA application, include `include` and link `lib\noMlp.lib`. Since the library contains CUDA device code, use `nvcc` (or configure CUDA device linking in Visual Studio) for the final application link. For example, with an application source named `app.cu`:

```powershell
nvcc -std=c++17 -rdc=true -Xcompiler "/std:c++17 /MD" -Iinclude app.cu lib\noMlp.lib -o app.exe
```

## Single-sample use

Set `network` to the node counts from input to output. `node_in` and `node_out` must match the first and last entries. Supported activations are `tanh`, `relu`, and `leakyrelu`.

```cpp
#include "noMlp.h"
#include <cstdio>
#include <vector>

int main()
{
    modelconfig config;
    config.node_in = 2;
    config.node_out = 1;
    config.lr = 0.01f;
    config.activation = activationType::tanh;
    config.network = {2, 4, 1};

    noMlp model;
    model.init(config);
    if (model.layers.empty())
        return 1;

    std::vector<float> input{0.25f, 0.75f};
    std::vector<float> target{1.0f};

    const std::vector<float> output = model.pridiction(input);
    if (output.empty()) {
        model.close();
        return 1;
    }

    const float loss = model.backward(target); // Updates weights for this sample.
    std::printf("prediction: %f, loss: %f\n", output[0], loss);

    model.close();
    return 0;
}
```

The method is currently spelled `pridiction` in the public API. For training, repeat prediction followed by `backward()` for each sample.

## Batched training

For batching, set `batching`, `datasize`, and `batchsize`. Input samples are concatenated in row order; targets use the same sample order. Each `train_batch()` call trains one epoch and returns the average loss.

```cpp
modelconfig config;
config.node_in = 2;
config.node_out = 1;
config.lr = 0.01f;

config.batchsize = 2;
config.datasize = 4;
config.activation = activationType::relu;
config.network = {2, 4, 1};

noMlp model;
model.init(config);

std::vector<float> inputs{
    0.0f, 0.0f,
    0.0f, 1.0f,
    1.0f, 0.0f,
    1.0f, 1.0f
};
std::vector<float> targets{0.0f, 1.0f, 1.0f, 0.0f};

const float loss = model.train_batch(inputs, targets);
model.close();
```

## Checkpoints

Set a checkpoint directory before saving or loading. Loading requires an initialized model with the same network structure.

```cpp
modelconfig config;
config.node_in = 2;
config.node_out = 1;
config.lr = 0.01f;
config.network = {2, 4, 1};
config.checkpoint_adrs = "checkpoints/demo";

noMlp model;
model.init(config);

model.save_checkpoint();
model.load_checkpoint();
model.close();
```

Errors are printed with the `nomlp:` prefix. Call `close()` when the model is no longer needed to release its CUDA buffers.
