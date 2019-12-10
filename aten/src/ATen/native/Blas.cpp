#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <ATen/native/Blas.h>
#include <ATen/NamedTensorUtils.h>
#include <ATen/core/EnableNamedTensor.h>

namespace at { namespace native {

DEFINE_DISPATCH(addmv_stub);

Tensor &addmv_out(Tensor& result, const Tensor &self_, const Tensor &mat, const Tensor &vec, Scalar beta, Scalar alpha) {
  { // scope of NoNamesGuard

  at::NoNamesGuard guard;
  auto result_sizes = std::vector<int64_t> {mat.size(0)};
  if (!result.defined()) {
    result = at::empty(result_sizes, mat.options());
  }
  if (result.sizes() != result_sizes) {
    result.resize_(result_sizes);
  }

  Tensor self = self_;
  if (self_.dim() == 0 || self_.size(0) == 1) {
    self = self_.expand({mat.size(0)});
  }

  TORCH_CHECK((mat.dim() == 2 && vec.dim() == 1 && self.dim() == 1),
    "vector + matrix @ vector expected, got ", self.dim(), ", ", mat.dim(), ", ", vec.dim());
  TORCH_CHECK((mat.size(1) == vec.size(0) && mat.size(0) == self.size(0)),
    "size mismatch, get ", self.size(0), ", ", mat.size(0), "x", mat.size(1), ",", vec.size(0));

  if (&result != &self) {
    at::native::copy_(result, self);
  }

  if (result.numel() != 0) {
    addmv_stub(self.device().type(), result, self, mat, vec, beta, alpha);
  }

  } // scope of NoNamesGuard
  at::namedinference::propagate_names_for_addmv(result, mat, vec, self_);
  return result;
}

Tensor addmv(const Tensor &self, const Tensor &mat, const Tensor &vec, Scalar beta, Scalar alpha) {
  Tensor result;
  return native::addmv_out(result, self, mat, vec, beta, alpha);
}

Tensor &addmv_(Tensor &self, const Tensor &mat, const Tensor &vec, Scalar beta, Scalar alpha) {
  return native::addmv_out(self, self, mat, vec, beta, alpha);
}

Tensor &mv_out(Tensor& result, const Tensor &self, const Tensor &vec) {
  return native::addmv_out(result, result, self, vec, 0, 1);
}

Tensor mv(const Tensor &self, const Tensor &vec) {
  Tensor result;
  return native::mv_out(result, self, vec);
}

}}  // namespace at::native
