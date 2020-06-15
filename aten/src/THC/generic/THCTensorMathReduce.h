#ifndef THC_GENERIC_FILE
#define THC_GENERIC_FILE "THC/generic/THCTensorMathReduce.h"
#else

#if !defined(THC_REAL_IS_BOOL)

#if defined(THC_REAL_IS_FLOAT) || defined(THC_REAL_IS_DOUBLE) || defined(THC_REAL_IS_HALF)

THC_API void THCTensor_(renorm)(THCState *state, THCTensor* self, THCTensor* src, scalar_t value, int dimension, scalar_t max_norm);

THC_API accreal THCTensor_(std_all)(THCState *state, THCTensor *self, bool unbiased);
THC_API accreal THCTensor_(var_all)(THCState *state, THCTensor *self, bool unbiased);

#endif

#endif

#endif
