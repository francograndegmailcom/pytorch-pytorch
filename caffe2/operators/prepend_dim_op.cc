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

#include "caffe2/operators/prepend_dim_op.h"

namespace caffe2 {

REGISTER_CPU_OPERATOR(PrependDim, PrependDimOp<CPUContext>);
REGISTER_CPU_OPERATOR(MergeDim, MergeDimOp<CPUContext>);

OPERATOR_SCHEMA(PrependDim)
    .NumInputs(1)
    .NumOutputs(1)
    .AllowInplace({{0, 0}})
    .SetDoc(R"DOC(
Reshape the tensor by prepending a dimension of fixed size and dividing the
size of the next dimension by that amount.
)DOC")
    .Arg("dim_size", "Size of the dimension to prepend.")
    .Input(0, "data", "An input tensor.")
    .Output(0, "reshaped", "Reshaped tensor.");

OPERATOR_SCHEMA(MergeDim)
    .NumInputs(1)
    .NumOutputs(1)
    .AllowInplace({{0, 0}})
    .SetDoc(R"DOC(
Merge first two dimensions in a single dimension with size dim(0) * dim(1).
)DOC")
    .Input(0, "data", "An input tensor.")
    .Output(0, "reshaped", "Reshaped tensor.");

class GetPrependDimGradient : public GradientMakerBase {
  using GradientMakerBase::GradientMakerBase;
  vector<OperatorDef> GetGradientDefs() override {
    return SingleGradientDef(
        "MergeDim", "", vector<string>{GO(0)}, vector<string>{GI(0)});
  }

  // Arguments are no longer needed in backprop.
  bool CopyArguments() const override {
    return false;
  }
};

REGISTER_GRADIENT(PrependDim, GetPrependDimGradient);

} // namespace caffe2
