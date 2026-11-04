#pragma once

#include "noMlp.h"
#include <cuda_runtime.h>

inline float beta1 = 0.9f;
inline float beta2 = 0.999f;
inline bool initadam(noMlp &m)
{
    m.adam_step = 1;
    if (!geterror("Adam weight allocation", cudaMalloc(&m.adam_weights, m.weightbuffersize * sizeof(float2)))) return false;
    if (!geterror("Adam bias allocation", cudaMalloc(&m.adam_bias, m.biasbuffersize * sizeof(float2)))) return false;

    if (!geterror("Adam weight initialization", cudaMemset(m.adam_weights, 0, m.weightbuffersize * sizeof(float2)))) return false;
    if (!geterror("Adam bias initialization", cudaMemset(m.adam_bias, 0, m.biasbuffersize * sizeof(float2)))) return false;
    return true;
}
inline void clearadam(noMlp &m)
{
    if (m.adam_weights != nullptr)
    {
        const cudaError_t err = cudaFree(m.adam_weights);
        geterror("cudaFree Adam weights", err);
        if (err == cudaSuccess)
            m.adam_weights = nullptr;
    }
    if (m.adam_bias != nullptr)
    {
        const cudaError_t err = cudaFree(m.adam_bias);
        geterror("cudaFree Adam bias", err);
        if (err == cudaSuccess)
            m.adam_bias = nullptr;
    }
    m.adam_step = 1;
}

inline __device__ void adam_update(float &w, float2 &mv, const float gradient,
                                   const float lrate, const float4 beta)
{
    float b1 = beta.x;
    float b2 = beta.y;
    float bt1 = beta.z;
    float bt2 = beta.w;
    const float safeGradient = fminf(fmaxf(gradient, -10.0f), 10.0f);
    const float m = b1 * mv.x + (1.0f - b1) * safeGradient;
    const float v = b2 * mv.y + (1.0f - b2) * safeGradient * safeGradient;
    mv.x = m;
    mv.y = v;
    w -= lrate * (m * bt1) / (sqrtf(v * bt2) + 1e-6f);
}
