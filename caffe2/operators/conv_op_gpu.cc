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

#include "caffe2/operators/conv_op.h"
#include "caffe2/operators/conv_op_impl.h"
#include "caffe2/core/context_gpu.h"

namespace caffe2 {
REGISTER_CUDA_OPERATOR(Conv, ConvOp<float, CUDAContext>);
REGISTER_CUDA_OPERATOR(ConvGradient, ConvGradientOp<float, CUDAContext>);

REGISTER_CUDA_OPERATOR(Conv1D, ConvOp<float, CUDAContext>);
REGISTER_CUDA_OPERATOR(Conv1DGradient, ConvGradientOp<float, CUDAContext>);

REGISTER_CUDA_OPERATOR(Conv2D, ConvOp<float, CUDAContext>);
REGISTER_CUDA_OPERATOR(Conv2DGradient, ConvGradientOp<float, CUDAContext>);

REGISTER_CUDA_OPERATOR(Conv3D, ConvOp<float, CUDAContext>);
REGISTER_CUDA_OPERATOR(Conv3DGradient, ConvGradientOp<float, CUDAContext>);
}  // namespace caffe2
