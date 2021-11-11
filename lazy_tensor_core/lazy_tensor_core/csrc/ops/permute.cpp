#include "lazy_tensor_core/csrc/ops/permute.h"

#include "lazy_tensor_core/csrc/ts_backend/ts_shape_inference.h"
#include "lazy_tensor_core/csrc/helpers.h"
#include "lazy_tensors/computation_client/util.h"
#include "lazy_tensors/shape_util.h"

namespace torch_lazy_tensors {
namespace ir {
namespace ops {

Permute::Permute(const torch::lazy::Value& input, std::vector<int64_t> dims)
    : TsNode(torch::lazy::OpKind(at::aten::permute), {input},
             /*num_outputs=*/1, torch::lazy::MHash(dims)),
      dims_(std::move(dims)) {
  SetShapeDeferred(
      [&]() { return compiler::InferShape(this); });
}

std::string Permute::ToString() const {
  std::stringstream ss;
  ss << TsNode::ToString() << ", dims=(" << c10::Join(", ", dims_) << ")";
  return ss.str();
}

torch::lazy::Shape Permute::MakePermuteShape(
    const torch::lazy::Shape& source_shape,
    c10::ArrayRef<int64_t> permutation) {
  return lazy_tensors::ShapeUtil::MakeShape(
      source_shape.scalar_type(),
      lazy_tensors::Permute(permutation, source_shape.sizes()));
}

}  // namespace ops
}  // namespace ir
}  // namespace torch_lazy_tensors
