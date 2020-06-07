#ifndef THC_GENERIC_FILE
#define THC_GENERIC_FILE "THCUNN/generic/LogSigmoid.cu"
#else

#include <THCUNN/common.h>

void THNN_(LogSigmoid_updateOutput)(
           THCState *state,
           THCTensor *input,
           THCTensor *output,
           THCTensor *buffer)
{
  THCUNN_assertSameGPU(state, 2, input, output);
  THCTensor_(resizeAs)(state, output, input);
  THC_pointwiseApply2<scalar_t, scalar_t>(state, output, input, logSigmoid_updateOutput_functor<scalar_t>());
}

#endif
