#include "lazy_tensor_core/csrc/helpers.h"

#include <c10/util/Half.h>

#include <limits>

#include "lazy_tensor_core/csrc/tensor_util.h"
#include "lazy_tensors/computation_client/sys_util.h"
#include "lazy_tensors/computation_client/util.h"
#include "lazy_tensors/shape_util.h"

namespace torch_lazy_tensors {

std::vector<int64_t> Helpers::DropDimensions(c10::ArrayRef<int64_t> sizes,
                                             c10::ArrayRef<int64_t> drop_dims) {
  std::vector<int64_t> new_dims;
  size_t drop_index = 0;
  for (size_t i = 0; i < sizes.size(); ++i) {
    if (drop_index < drop_dims.size() && i == drop_dims[drop_index]) {
      ++drop_index;
    } else {
      new_dims.push_back(sizes[i]);
    }
  }
  CHECK_EQ(drop_index, drop_dims.size());
  return new_dims;
}

int64_t Helpers::GetCanonicalDimensionIndex(int64_t dim, int64_t rank) {
  int64_t min_shape_dim = -rank;
  int64_t max_shape_dim = rank - 1;
  CHECK(min_shape_dim <= dim && dim <= max_shape_dim)
      << "Value out of range (expected to be in range of [" << min_shape_dim
      << ", " << max_shape_dim << "], but got " << dim << ")";
  int64_t dim_index = dim < 0 ? rank + dim : dim;
  CHECK_GE(dim_index, 0);
  CHECK_LT(dim_index, rank);
  return dim_index;
}

std::vector<int64_t> Helpers::GetCanonicalDimensionIndices(
    c10::ArrayRef<int64_t> dimensions, int64_t rank) {
  std::vector<int64_t> canonical_dim_indices;
  for (int64_t dim : dimensions) {
    canonical_dim_indices.push_back(GetCanonicalDimensionIndex(dim, rank));
  }
  return canonical_dim_indices;
}

int64_t Helpers::GetCanonicalPosition(c10::ArrayRef<int64_t> dimensions,
                                      int64_t dim, int64_t pos) {
  dim = GetCanonicalDimensionIndex(dim, dimensions.size());
  if (pos < 0) {
    pos = GetCanonicalDimensionIndex(pos, dimensions[dim]);
  } else {
    pos = std::min<int64_t>(pos, dimensions[dim]);
  }
  return pos;
}

std::vector<int64_t> Helpers::MakeTransposePermutation(int64_t dim0,
                                                       int64_t dim1,
                                                       int64_t rank) {
  int64_t canonical_dim0 = GetCanonicalDimensionIndex(dim0, rank);
  int64_t canonical_dim1 = GetCanonicalDimensionIndex(dim1, rank);
  auto permute_dims = lazy_tensors::util::Iota<int64_t>(rank);
  std::swap(permute_dims[canonical_dim0], permute_dims[canonical_dim1]);
  return permute_dims;
}

std::vector<int64_t> Helpers::GetPromotedShape(
    c10::ArrayRef<int64_t> shape1_dims, c10::ArrayRef<int64_t> shape2_dims) {
  std::vector<int64_t> dimensions;
  // If the rank of a shape is bigger than then other, fill up the first
  // dimensions with the ones of the bigger.
  // Example:
  //   shape1 = [9, 7, 6, 5, 2]
  //   shape2 =       [6, 1, 2]
  // Insert [9, 7] into the dimensions vector.
  if (shape1_dims.size() > shape2_dims.size()) {
    dimensions.insert(
        dimensions.end(), shape1_dims.begin(),
        shape1_dims.begin() + (shape1_dims.size() - shape2_dims.size()));
  } else if (shape2_dims.size() > shape1_dims.size()) {
    dimensions.insert(
        dimensions.end(), shape2_dims.begin(),
        shape2_dims.begin() + (shape2_dims.size() - shape1_dims.size()));
  }
  // For the common dimensions, they must match, or one of them be 1.
  size_t min_size = std::min(shape1_dims.size(), shape2_dims.size());
  for (int64_t i = 0; i < min_size; ++i) {
    int64_t dim1 = shape1_dims[shape1_dims.size() - min_size + i];
    int64_t dim2 = shape2_dims[shape2_dims.size() - min_size + i];
    CHECK(dim1 == dim2 || dim1 == 1 || dim2 == 1)
        << "(" << c10::Join(", ", shape1_dims) << ") and ("
        << c10::Join(", ", shape1_dims) << ")";
    if (dim1 == 0 || dim2 == 0) {
      dimensions.push_back(0);
    } else {
      dimensions.push_back(std::max<int64_t>(dim1, dim2));
    }
  }
  return dimensions;
}

torch::lazy::Shape Helpers::GetPromotedShape(
    const torch::lazy::Shape& shape1, const torch::lazy::Shape& shape2) {
  return lazy_tensors::ShapeUtil::MakeShape(
      shape1.scalar_type(),
      GetPromotedShape(shape1.sizes(), shape2.sizes()));
}

torch::lazy::Shape Helpers::GetPromotedBinaryOpShape(
    const torch::lazy::Shape& shape1, const torch::lazy::Shape& shape2) {
  return lazy_tensors::ShapeUtil::MakeShape(
      promoteTypes(shape1.scalar_type(), shape2.scalar_type()),
      GetPromotedShape(shape1.sizes(), shape2.sizes()));
}

}  // namespace torch_lazy_tensors
