#pragma once

#include <cuda_runtime.h>


inline __device__ float activation(float x ,int type){
  
    switch (type){
       case 1: // Tanh
        return tanhf(x);

    case 2: // ReLU
        return x > 0.0f ? x : 0.0f;

    case 3: // Leaky ReLU
        return x > 0.0f ? x : 0.01f * x;

    default:
        return x; 
    }
}

inline __device__ float dslope(float x, int type)
{
	switch (type)
	{
	case 1: // Tanh
	{
		const float t = tanhf(x);
		return 1.0f - t * t;
	}
	case 2: // ReLU
		return x > 0.0f ? 1.0f : 0.0f;
	case 3: // LeakyReLU
		return x > 0.0f ? 1.0f : 0.01f;
	default:
		return 1.0f;
	}
}
