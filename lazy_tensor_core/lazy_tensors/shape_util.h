#pragma once

#include <c10/util/Optional.h>
#include <c10/core/Scalar.h>
#include <torch/csrc/lazy/core/shape.h>

#include <complex>

#include "lazy_tensors/computation_client/util.h"
#include "torch/csrc/jit/tensorexpr/types.h"
#include "torch/csrc/lazy/core/hash.h"

namespace torch {
namespace lazy {
    // Adapters that provide torch::lazy Hash functions for lazy_tensors types
    hash_t Hash(const torch::lazy::Shape& shape);
}
}

namespace lazy_tensors {

class ShapeIndex {
 public:
  ShapeIndex() = default;
  ShapeIndex(std::initializer_list<int64_t> init) : indices_(init) {}

  bool empty() const { return indices_.empty(); }
  size_t size() const { return indices_.size(); }
  void push_back(int64_t value) { indices_.push_back(value); }
  void pop_back() { indices_.pop_back(); }

  const int64_t& operator[](size_t i) const { return indices_[i]; }
  int64_t& operator[](size_t i) { return indices_[i]; }

 private:
  std::vector<int64_t> indices_;
};

class ShapeUtil {
 public:
  static int64_t ElementsIn(const torch::lazy::Shape& shape) {
    return util::Multiply<int64_t>(shape.sizes());
  }
  static bool SameDimensions(const torch::lazy::Shape& lhs, const torch::lazy::Shape& rhs) {
    return lhs.sizes() == rhs.sizes();
  }

  static bool Compatible(const torch::lazy::Shape& lhs, const torch::lazy::Shape& rhs) {
    return lhs == rhs;
  }

  static torch::lazy::Shape MakeShape(c10::ScalarType element_type,
                         c10::ArrayRef<int64_t> dimensions) {
    return torch::lazy::Shape(element_type, dimensions);
  }

  static bool ElementIsIntegral(const torch::lazy::Shape& shape) {
    return isIntegralType(shape.scalar_type(), /* include_bool */ true);
  }

  // Compute a hash for `shape`.
  static size_t Hash(const torch::lazy::Shape& shape);
};

}  // namespace lazy_tensors
