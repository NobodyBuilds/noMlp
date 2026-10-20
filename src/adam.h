#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>

inline float2* adam_weights=nullptr;//x= m,y=v;
inline float2* adam_bias=nullptr;//x= m,y=v;
inline int adam_step=1;
inline void initadam(int wn,int bn){
    geterror("adam weight malloc",cudaMalloc(&adam_weights,wn*sizeof(float2)));
    geterror("adam bias malloc",cudaMalloc(&adam_bias,bn*sizeof(float2)));

    geterror("adam weight memset",cudaMemset(adam_weights,0,wn*sizeof(float2)));
    geterror("adam bias memset",cudaMemset(adam_bias,0,bn*sizeof(float2)));
}
inline void clearadam(){
    if (adam_weights != nullptr) {
        const cudaError_t err = cudaFree(adam_weights);
        geterror("cudaFree Adam weights", err);
        if (err == cudaSuccess) adam_weights = nullptr;
    }
    if (adam_bias != nullptr) {
        const cudaError_t err = cudaFree(adam_bias);
        geterror("cudaFree Adam bias", err);
        if (err == cudaSuccess) adam_bias = nullptr;
    }
    adam_step = 1;
}



inline __device__ void adam_update(float& w, float2& mv, const float gradient,
	const float lrate, const float4 beta)
{
	float b1=beta.x;
	float b2=beta.y;
	float bt1=beta.z;
	float bt2=beta.w;
	const float safeGradient = fminf(fmaxf(gradient, -10.0f), 10.0f);
	const float m = b1 * mv.x + (1.0f - b1) * safeGradient;
	const float v = b2 * mv.y + (1.0f - b2) * safeGradient * safeGradient;
	mv.x = m; mv.y = v;
	w -= lrate * (m * bt1) / (sqrtf(v * bt2) + 1e-6f);      
}
