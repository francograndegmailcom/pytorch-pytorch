#include "THCTensorMath.h"
#include "THCGeneral.h"
#include "TH/THHalf.h"
#include "THCTensorCopy.h"
#include "THCApply.cuh"
#include "THCNumerics.cuh"
#include "THCTensorMathCompareT.cuh"
#include "THCTensor.hpp"

template <typename T>
struct TensorAddConstantOp {
  TensorAddConstantOp(T v) : val(v) {}
  __device__ __forceinline__ void operator()(T* out, T* in) {
    *out = *in + val;
  }

  __device__ __forceinline__ void operator()(T* v) {
    *v += val;
  }

  const T val;
};

template <typename T>
struct TensorSubConstantOp {
  TensorSubConstantOp(T v) : val(v) {}
  __device__ __forceinline__ void operator()(T* out, T* in) {
    *out = *in - val;
  }

  __device__ __forceinline__ void operator()(T* v) {
    *v -= val;
  }

  const T val;
};

template <typename T>
struct TensorMulConstantOp {
  TensorMulConstantOp(T v) : val(v) {}
  __device__ __forceinline__ void operator()(T* out, T* in) {
    *out = *in * val;
  }

  __device__ __forceinline__ void operator()(T* v) {
    *v *= val;
  }

  const T val;
};

template <typename T>
struct TensorDivConstantOp {
  TensorDivConstantOp(T v) : val(v) {}
  __device__ __forceinline__ void operator()(T* out, T* in) {
    *out = *in / val;
  }

  __device__ __forceinline__ void operator()(T* v) {
    *v /= val;
  }

  const T val;
};

template <>
struct TensorDivConstantOp<float> {
  TensorDivConstantOp(float v) : val(1.f / v) {}
  __device__ __forceinline__ void operator()(float* out, float* in) {
    *out = *in * val;
  }

  __device__ __forceinline__ void operator()(float* v) {
    *v *= val;
  }

  const float val;
};

template <>
struct TensorDivConstantOp<double> {
  TensorDivConstantOp(double v) : val(1. / v) {}
  __device__ __forceinline__ void operator()(double* out, double* in) {
    *out = *in * val;
  }

  __device__ __forceinline__ void operator()(double* v) {
    *v *= val;
  }

  const double val;
};

template<typename T>
static __device__ __forceinline__
typename std::enable_if<std::is_signed<T>::value, bool>::type
modulo_wrap(T a, T b) {
  return (a != 0) && (a < 0) != (b < 0);
}

template<typename T>
static __device__ __forceinline__
typename std::enable_if<std::is_unsigned<T>::value, bool>::type
modulo_wrap(T a, T b) {
  return false;
}

template <typename T>
struct TensorRemainderOp {
  TensorRemainderOp(T v) : val(v) {}
  __device__ __forceinline__ void operator()(T* out, T* in) {
    *out = *in % val;
    if (modulo_wrap<T>(*out, val)) {
      *out += val;
    }
  }

  __device__ __forceinline__ void operator()(T* v) {
    *v = *v % val;
    if (modulo_wrap<T>(*v, val)) {
      *v += val;
    }
  }

  const T val;
};

template <>
struct TensorRemainderOp<float> {
  TensorRemainderOp(float v) : val(v) {}
  __device__ __forceinline__ void operator()(float* out, float* in) {
    *out = *in - val * floorf(*in / val);
  }

  __device__ __forceinline__ void operator()(float* v) {
    *v = *v - val * floorf(*v / val);
  }

  const float val;
};

template <>
struct TensorRemainderOp<double> {
  TensorRemainderOp(double v) : val(v) {}
  __device__ __forceinline__ void operator()(double* out, double* in) {
    *out = *in - val * floor(*in / val);
  }

  __device__ __forceinline__ void operator()(double* v) {
    *v = *v - val * floor(*v / val);
  }

  const double val;
};

template <>
struct TensorRemainderOp<THHalf> {
  TensorRemainderOp(THHalf v): val(v) {}

  __device__ __forceinline__ void operator()(THHalf* out, THHalf* in) {
    *out = *in - val * floorf(*in / val);
  }

  __device__ __forceinline__ void operator()(THHalf* v) {
    *v = *v - val * floorf(*v / val);
  }

  const THHalf val;
};

template <typename T>
struct TensorFmodOp {
  TensorFmodOp(T v) : val((float)v) {}
  __device__ __forceinline__ void operator()(T* out, T* in) {
    *out = (T) fmodf((float) *in, val);
  }

  __device__ __forceinline__ void operator()(T* v) {
    *v = (T) fmodf((float) *v, val);
  }

  const float val;
};

template <>
struct TensorFmodOp<double> {
  TensorFmodOp(double v) : val(v) {}
  __device__ __forceinline__ void operator()(double* out, double* in) {
    *out = fmod(*in, val);
  }

  __device__ __forceinline__ void operator()(double* v) {
    *v = fmod(*v, val);
  }

  const double val;
};

template <typename T, int Upper>
struct TensorTriOp {
  TensorTriOp(T *start_, int64_t stride0_, int64_t stride1_, int64_t k_)
    : start(start_), stride0(stride0_), stride1(stride1_), k(k_) {}

  __device__ __forceinline__ int mask(T *out) {
    ptrdiff_t n = out - start;
    int64_t row, col;
    if (stride0 > stride1)
    {
      row = (int64_t) (n / stride0);
      col = (int64_t) ((n % stride0) / stride1);
    }
    else
    {
      row = (int64_t) ((n % stride1) / stride0);
      col = (int64_t) (n / stride1);
    }

    return Upper ? (col - row >= k) : (col - row <= k);
  }

  __device__ __forceinline__ void operator()(T* out, T* in) {
    *out = mask(out) ? *in : ScalarConvert<int, T>::to(0);
  }

  __device__ __forceinline__ void operator()(T* v) {
    if (!mask(v))
      *v = ScalarConvert<int, T>::to(0);
  }

  const T *start;
  const int64_t stride0, stride1, k;
};

template <typename T>
struct TensorLShiftConstantOp {
  TensorLShiftConstantOp(T v) : val(v) {}
  __device__ __forceinline__ void operator()(T* out, T* in) {
    *out = *in << val;
  }

  __device__ __forceinline__ void operator()(T* v) {
    *v <<= val;
  }

  const T val;
};

template <typename T>
struct TensorRShiftConstantOp {
  TensorRShiftConstantOp(T v) : val(v) {}
  __device__ __forceinline__ void operator()(T* out, T* in) {
    *out = *in >> val;
  }

  __device__ __forceinline__ void operator()(T* v) {
    *v >>= val;
  }

  const T val;
};

template <typename T>
struct TensorBitAndConstantOp {
  TensorBitAndConstantOp(T v) : val(v) {}
  __device__ __forceinline__ void operator()(T* out, T* in) {
    *out = *in & val;
  }

  __device__ __forceinline__ void operator()(T* v) {
    *v &= val;
  }

  const T val;
};

template <typename T>
struct TensorBitOrConstantOp {
  TensorBitOrConstantOp(T v) : val(v) {}
  __device__ __forceinline__ void operator()(T* out, T* in) {
    *out = *in | val;
  }

  __device__ __forceinline__ void operator()(T* v) {
    *v |= val;
  }

  const T val;
};

template <typename T>
struct TensorBitXorConstantOp {
  TensorBitXorConstantOp(T v) : val(v) {}
  __device__ __forceinline__ void operator()(T* out, T* in) {
    *out = *in ^ val;
  }

  __device__ __forceinline__ void operator()(T* v) {
    *v ^= val;
  }

  const T val;
};

#include "generic/THCTensorMathPairwise.cu"
#include "THCGenerateAllTypes.h"
