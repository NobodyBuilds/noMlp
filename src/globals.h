#pragma once
#include <cuda_runtime.h>

// Shared CUDA launch helpers.
inline int blocks(int n, int thread)
{
    return (n + thread - 1) / thread;
}
