#include <THCUNN/THCUNN.h>
#include <TH/THHalf.h>
#include <THCUNN/THCHalfAutoNumerics.cuh>
#include <THC/THCApply.cuh>

template <typename T>
struct logSigmoid_updateOutput_functor
{
  __device__ void operator()(T *output, const T *input) const {
    const T max = fmaxType(T{0}, -*input);
    const T z = ::exp(-max) + ::exp(-*input -max);
    *output = -(max + static_cast<T>(std::log(z)));
  }
};


template <>
struct logSigmoid_updateOutput_functor<half> {
  __device__ __forceinline__ void operator()(half* output, const half *input) const {
    float in = __half2float(*input);
    float max = fmaxType(0.f, -in);
    float z = ::exp(-max) + ::exp(-in - max);
    *output = __float2half(-(max + std::log(z)));
  }
};

#include <THCUNN/generic/LogSigmoid.cu>
#include <THC/THCGenerateFloatTypes.h>
