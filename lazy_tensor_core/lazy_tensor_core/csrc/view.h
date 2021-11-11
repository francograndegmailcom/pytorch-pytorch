#pragma once

#include <c10/util/Optional.h>
#include <torch/csrc/lazy/core/shape.h>

#include <memory>
#include <vector>

#include "torch/csrc/lazy/core/ir.h"

namespace torch_lazy_tensors {

struct SelectInfo {
  bool operator==(const SelectInfo& ref) const {
    return dim == ref.dim && start == ref.start && end == ref.end &&
           stride == ref.stride;
  }

  int64_t dim = 0;
  int64_t start = 0;
  int64_t end = 0;
  int64_t stride = 0;
};

struct AsStridedInfo {
  bool operator==(const AsStridedInfo& ref) const {
    return offset == ref.offset && stride == ref.stride;
  }

  std::vector<int64_t> stride;
  int64_t offset = 0;
};

struct DiagonalInfo {
  bool operator==(const DiagonalInfo& ref) const {
    return offset == ref.offset && dim1 == ref.dim1 && dim2 == ref.dim2;
  }

  int64_t offset = 0;
  int64_t dim1 = 0;
  int64_t dim2 = 1;
};

struct ViewInfo {
  enum class Type {
    kInvalid,
    kNarrow,
    kNoOp,
    kPermute,
    kReshape,
    kResize,
    kSelect,
    kAsStrided,
    kDiagonal,
  };

  ViewInfo() = default;
  ViewInfo(Type view_type, torch::lazy::Shape shape,
           torch::lazy::Shape source_shape);
  ViewInfo(Type view_type, torch::lazy::Shape source_shape,
           std::vector<int64_t> permutation);
  ViewInfo(Type view_type, const torch::lazy::Shape& source_shape,
           SelectInfo select);
  ViewInfo(Type view_type, torch::lazy::Shape shape,
           torch::lazy::Shape source_shape, AsStridedInfo as_strided);
  ViewInfo(Type view_type, const torch::lazy::Shape& source_shape,
           DiagonalInfo diagonal);

  bool operator==(const ViewInfo& ref) const {
    return view_type == ref.view_type && shape == ref.shape &&
           indices == ref.indices && source_shape == ref.source_shape &&
           permutation == ref.permutation && select == ref.select &&
           as_strided == ref.as_strided && diagonal == ref.diagonal;
  }

  Type view_type = Type::kInvalid;
  // The shape of the result of a view. In case of narrowing, this represents
  // the size of the narrow slice.
  torch::lazy::Shape shape;
  // In case of narrowing, the starting indices from where the narrow slice is
  // cut.
  std::vector<int64_t> indices;
  // The shape of the source of this view.
  torch::lazy::Shape source_shape;
  // The permutation to be used. If empty, this is not a permute operation.
  std::vector<int64_t> permutation;
  // Information used for sliced views.
  c10::optional<SelectInfo> select;
  // Information used for as_strided views.
  c10::optional<AsStridedInfo> as_strided;
  // Information used for diagonal views.
  c10::optional<DiagonalInfo> diagonal;
};

// When a "view" (capture by reference) is taken on a node, an Alias object is
// created on the captured node itself, with its current IR Node value.
class Alias {
 public:
  struct UpdateData {
    torch::lazy::Value ir_value;
    std::vector<ViewInfo> view_infos;
  };

  explicit Alias(torch::lazy::Value ir_value)
      : root_ir_value_(std::move(ir_value)) {}

  size_t generation() const { return generation_; }

  // Appends an update to the IR value stored within the alias. The ir_value is
  // the value to be written, and view_infos represents the forward path from
  // the alias's ir_value to the update ir_value.
  void Update(torch::lazy::Value ir_value, std::vector<ViewInfo> view_infos);

  torch::lazy::Value SyncUpdateOperations();

 private:
  // The IR value which is the root at which the view was created.
  torch::lazy::Value root_ir_value_;
  // The stacked updates on the view. Orders matter, as most recent updates
  // might overwrite older ones.
  std::vector<UpdateData> updates_;
  // Incremented every time an update happens. Used by view to track alias
  // changes and regenerate the most current value.
  size_t generation_ = 0;
};

class View {
 public:
  View(torch::lazy::Shape shape, std::shared_ptr<Alias> alias,
       ViewInfo view_info);
  View(torch::lazy::Shape shape, std::shared_ptr<Alias> alias,
       std::vector<ViewInfo> view_infos);

  void Update(torch::lazy::Value ir_value);

  const torch::lazy::Shape& shape() const { return shape_; }

  const std::shared_ptr<Alias>& alias() const { return alias_; }

  std::shared_ptr<View> CreateSubView(torch::lazy::Shape shape,
                                      ViewInfo view_info);

  // Extracts the current IrNode out of a view, into a IrNode structure
  // where the updated fields tells whether a new IR value has been created, or
  // the cached one returned.
  std::tuple<torch::lazy::Value, bool> GetViewIrNode();

  bool IsUpToDate() const {
    return ir_value_ && generation_ == alias_->generation();
  }

 private:
  std::vector<ViewInfo> view_infos_;
  torch::lazy::Shape shape_;
  std::shared_ptr<Alias> alias_;
  torch::lazy::Value ir_value_;
  size_t generation_ = 0;
};

}  // namespace torch_lazy_tensors
