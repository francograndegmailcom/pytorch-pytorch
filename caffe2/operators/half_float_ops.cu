/**
 * Copyright (c) 2016-present, Facebook, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "caffe2/operators/half_float_ops.h"

#include "caffe2/core/context_gpu.h"

#ifdef CAFFE_HAS_CUDA_FP16

namespace caffe2 {
namespace {
__global__ void FloatToHalfKernel(const int N, const float* X, half* Y) {
  CUDA_1D_KERNEL_LOOP(i, N) {
    Y[i] = __float2half(X[i]);
  }
}

__global__ void HalfToFloatKernel(const int N, const half* X, float* Y) {
  CUDA_1D_KERNEL_LOOP(i, N) {
    Y[i] = __half2float(X[i]);
  }
}
}

template <>
bool FloatToHalfOp<CUDAContext>::RunOnDevice() {
  auto& X = Input(0);
  auto* Y = Output(0);
  Y->ResizeLike(X);
  FloatToHalfKernel<<<
      CAFFE_GET_BLOCKS(X.size()),
      CAFFE_CUDA_NUM_THREADS,
      0,
      context_.cuda_stream()>>>(
      X.size(),
      X.data<float>(),
      reinterpret_cast<half*>(Y->mutable_data<float16>()));
  return true;
}

template <>
bool HalfToFloatOp<CUDAContext>::RunOnDevice() {
  auto& X = Input(0);
  auto* Y = Output(0);
  Y->ResizeLike(X);
  HalfToFloatKernel<<<
      CAFFE_GET_BLOCKS(X.size()),
      CAFFE_CUDA_NUM_THREADS,
      0,
      context_.cuda_stream()>>>(
      X.size(),
      reinterpret_cast<const half*>(X.data<float16>()),
      Y->mutable_data<float>());
  return true;
}

REGISTER_CUDA_OPERATOR(FloatToHalf, FloatToHalfOp<CUDAContext>);
REGISTER_CUDA_OPERATOR(HalfToFloat, HalfToFloatOp<CUDAContext>);
} // namespace caffe2

#endif // CAFFE_HAS_CUDA_FP16
