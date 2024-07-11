#define TORCH_ASSERT_NO_OPERATORS

#include <limits>

#include <ATen/native/Sorting.h>
#include <ATen/core/TensorBase.h>
#include <ATen/Dispatch.h>
#include <ATen/Dispatch_v2.h>
#include <ATen/Parallel.h>
#include <ATen/NumericUtils.h>
#include <ATen/TensorIterator.h>
#include <ATen/cpu/vec/functional.h>
#include <ATen/cpu/vec/vec.h>
#include <ATen/native/StridedRandomAccessor.h>
#include <ATen/native/CompositeRandomAccessor.h>
#include <ATen/native/TopKImpl.h>
#include <c10/core/WrapDimMinimal.h>
#include <c10/util/irange.h>

#ifdef USE_FBGEMM
#include <fbgemm/Utils.h>
#endif

#if USE_X86_SIMD_SORT && (defined(CPU_CAPABILITY_AVX512) || defined(CPU_CAPABILITY_AVX2))
#define XSS_COMPILE_TIME_SUPPORTED
#define XSS_USE_OPENMP
#include <src/x86simdsort-static-incl.h>
#endif

namespace at::native {

namespace {

template <typename func_t>
void _dim_apply(
    const TensorBase &values,
    const TensorBase &indices,
    int64_t dim,
    const std::string& method_name,
    const func_t& f) {
  auto iter = TensorIteratorConfig()
    .check_all_same_dtype(false)
    .resize_outputs(false)
    .declare_static_shape(values.sizes(), /*squash_dims=*/dim)
    .add_output(values)
    .add_output(indices)
    .build();

  auto values_dim_stride = values.stride(dim);
  auto indices_dim_stride = indices.stride(dim);
  auto dim_size = values.size(dim);

  AT_DISPATCH_V2(
    iter.dtype(), "sorting_kernel_method_name", AT_WRAP([&] {
      auto loop = [&](char** data, const int64_t* strides, int64_t n) {
        auto* values_data_bytes = data[0];
        auto* indices_data_bytes = data[1];

        if(values_data_bytes==nullptr || indices_data_bytes==nullptr){
          return;
        }

        for (const auto i C10_UNUSED : c10::irange(n)) {
          f(
            reinterpret_cast<scalar_t*>(values_data_bytes),
            values_dim_stride,
            reinterpret_cast<int64_t*>(indices_data_bytes),
            indices_dim_stride,
            dim_size
          );

          values_data_bytes += strides[0];
          indices_data_bytes += strides[1];
        }
      };

      int64_t grain_size = internal::GRAIN_SIZE / std::max(int64_t{1}, dim_size);
      iter.for_each(loop, /*grain_size=*/grain_size);
    }), kBool, kHalf, kBFloat16, AT_EXPAND(AT_ALL_TYPES), AT_EXPAND(AT_BAREBONES_UNSIGNED_TYPES)
  );
}

template <typename scalar_t>
struct KeyValueCompAsc {
  template <typename LHS, typename RHS>
  constexpr bool operator()(LHS lhs, RHS rhs) const {
    return (!_isnan<scalar_t>(get<0>(lhs)) && _isnan<scalar_t>(get<0>(rhs)))
      || (get<0>(lhs) < get<0>(rhs));
  }
};

template <typename scalar_t>
struct KeyValueCompDesc {
  template <typename LHS, typename RHS>
  constexpr bool operator()(LHS lhs, RHS rhs) const {
    return (_isnan<scalar_t>(get<0>(lhs)) && !_isnan<scalar_t>(get<0>(rhs)))
      || (get<0>(lhs) > get<0>(rhs));
  }
};

#ifdef USE_FBGEMM
static bool can_use_radix_sort(const TensorBase& values, const bool descending) {
  // radix_sort can be used only for 1D data
  if (values.dim() != 1) return false;
  // radix_sort sorts in ascending order
  if (descending) return false;
  // radix_sort works for integer values
  if (!at::isIntegralType(values.scalar_type(), /*includeBool=*/false)) return false;
  // performance improvements are visible for bigger tensor sizes, when radix_sort
  // is accelerated with OpenMP
  if (values.numel() < at::internal::GRAIN_SIZE || !fbgemm::is_radix_sort_accelerated_with_openmp()) return false;
  // TODO(DamianSzwichtenberg): radix_sort is a stable sorting algorithm,
  // should we check here, whether stable is set to true?

  return true;
}

static void parallel_sort1d_kernel(
    const TensorBase& values,
    const TensorBase& indices) {
  AT_DISPATCH_INTEGRAL_TYPES(values.scalar_type(), "parallel_sort1d_kernel", [&] {
    const auto elements = values.numel();
    auto* const keys = values.data_ptr<scalar_t>();
    auto* const vals = indices.data_ptr<int64_t>();
    std::vector<scalar_t> tmp_keys(elements);
    std::vector<int64_t> tmp_vals(elements);
    const scalar_t* sorted_keys = nullptr;
    const int64_t* sorted_vals = nullptr;

    std::tie(sorted_keys, sorted_vals) = fbgemm::radix_sort_parallel(
        keys,
        vals,
        tmp_keys.data(),
        tmp_vals.data(),
        elements,
        std::numeric_limits<scalar_t>::max(),
        values.scalar_type() != ScalarType::Byte);

    const bool sorted_in_place = keys == sorted_keys;
    if (!sorted_in_place) {
      const auto num_threads = at::get_num_threads();
      at::parallel_for(0, elements, elements / num_threads, [&](int64_t begin, int64_t end) {
        const auto job_size = end - begin;
        vec::map([](vec::Vectorized<scalar_t> x) -> vec::Vectorized<scalar_t> { return x; }, keys + begin, sorted_keys + begin, job_size);
        vec::map([](vec::Vectorized<int64_t> x) -> vec::Vectorized<int64_t> { return x; }, vals + begin, sorted_vals + begin, job_size);
      });
    }
  });
}
#endif

#if defined(XSS_COMPILE_TIME_SUPPORTED)

#define AT_DISPATCH_CASE_XSS_TYPES(...)          \
  AT_DISPATCH_CASE(at::ScalarType::Long, __VA_ARGS__) \
  AT_DISPATCH_CASE(at::ScalarType::Int, __VA_ARGS__) \
  AT_DISPATCH_CASE(at::ScalarType::Double, __VA_ARGS__) \
  AT_DISPATCH_CASE(at::ScalarType::Float, __VA_ARGS__)

#define AT_DISPATCH_XSS_TYPES(TYPE, NAME, ...) \
  AT_DISPATCH_SWITCH(TYPE, NAME, AT_DISPATCH_CASE_XSS_TYPES(__VA_ARGS__))

static bool can_use_xss_sort(const TensorBase& values, const TensorBase& indices, int64_t dim, const bool stable) {
  // xss_sort is not a stable sort
  if (stable) return false;

  auto type = values.scalar_type();
  if (! (type == ScalarType::Long || type == ScalarType::Int || type == ScalarType::Double || type == ScalarType::Float)) return false;

  return true;
}

static void xss_sort_kernel(
    const TensorBase& values,
    const TensorBase& indices,
    int64_t dim,
    bool descending) {
  auto iter = TensorIteratorConfig()
    .check_all_same_dtype(false)
    .resize_outputs(false)
    .declare_static_shape(values.sizes(), /*squash_dims=*/dim)
    .add_output(values)
    .add_output(indices)
    .build();

  using index_t = int64_t;

  AT_DISPATCH_XSS_TYPES(values.scalar_type(), "xss_sort_kernel", [&] {

    auto values_dim_stride = values.stride(dim);
    auto indices_dim_stride = indices.stride(dim);
    auto dim_size = values.size(dim);

    auto loop = [&](char** data, const int64_t* strides, int64_t n) {
      auto* values_data_bytes = data[0];
      auto* indices_data_bytes = data[1];

      if(values_data_bytes==nullptr || indices_data_bytes==nullptr){
        return;
      }

      if (values_dim_stride == 1 && indices_dim_stride == 1){
        for (const auto i C10_UNUSED : c10::irange(n)) {
          x86simdsortStatic::keyvalue_qsort<scalar_t, index_t>(
              reinterpret_cast<scalar_t*>(values_data_bytes),
              reinterpret_cast<index_t*>(indices_data_bytes),
              dim_size,
              true,
              descending);

          values_data_bytes += strides[0];
          indices_data_bytes += strides[1];
        }
      }else{
        std::vector<scalar_t> tmp_values(dim_size);
        std::vector<index_t> tmp_indices(dim_size);

        for (const auto i : c10::irange(n)) {
          TensorAccessor<scalar_t, 1> mode_values_acc(
              reinterpret_cast<scalar_t*>(data[0] + i * strides[0]),
              &dim_size, &values_dim_stride);
          TensorAccessor<index_t, 1> mode_indices_acc(
              reinterpret_cast<index_t*>(data[1] + i * strides[1]),
              &dim_size, &indices_dim_stride);

          for (const auto j : c10::irange(dim_size)) {
            tmp_values[j] = mode_values_acc[j];
            tmp_indices[j] = j;
          }

          x86simdsortStatic::keyvalue_qsort<scalar_t, index_t>(
              tmp_values.data(),
              tmp_indices.data(),
              dim_size,
              true,
              descending);

          for (const auto j : c10::irange(dim_size)) {
            mode_values_acc[j] = tmp_values[j];
            mode_indices_acc[j] = tmp_indices[j];
          }
        }
      }
    };

    int64_t grain_size = internal::GRAIN_SIZE / std::max(int64_t{1}, dim_size);
    iter.for_each(loop, /*grain_size=*/grain_size);

  });
}
#endif

static void sort_kernel(
    const TensorBase& self,
    const TensorBase& values,
    const TensorBase& indices,
    int64_t dim,
    bool descending,
    bool stable) {
  dim = maybe_wrap_dim(dim, values.dim());
  _fill_indices(indices, dim);
  if (self.stride(dim) == 0) {
    // check if stride is zero
    // https://github.com/pytorch/pytorch/issues/91420
    return;
  }

#if defined(XSS_COMPILE_TIME_SUPPORTED)
  if (can_use_xss_sort(values, indices, dim, stable)){
    xss_sort_kernel(values, indices, dim, descending);
    return;
  }
#endif

#ifdef USE_FBGEMM
  if (can_use_radix_sort(values, descending)) {
    parallel_sort1d_kernel(values, indices);
    return;
  }
#endif
  _dim_apply(
    values, indices, dim,
    "sort_cpu", [&](
      auto* values, int64_t values_dim_stride,
      auto* indices, int64_t indices_dim_stride,
      int64_t dim_size
    ) {
      using scalar_t = typename std::remove_pointer<decltype(values)>::type;
      auto values_accessor = StridedRandomAccessor<scalar_t>(
        values, values_dim_stride);
      auto indices_accessor = StridedRandomAccessor<int64_t>(
        indices, indices_dim_stride);
      auto composite_accessor = CompositeRandomAccessorCPU<
        decltype(values_accessor), decltype(indices_accessor)
      >(values_accessor, indices_accessor);

      if (descending) {
        if (stable) {
          std::stable_sort(composite_accessor, composite_accessor + dim_size,
            KeyValueCompDesc<scalar_t>());
        }
        else {
          std::sort(composite_accessor, composite_accessor + dim_size,
            KeyValueCompDesc<scalar_t>());
        }
      }
      else {
        if (stable) {
          std::stable_sort(composite_accessor, composite_accessor + dim_size,
            KeyValueCompAsc<scalar_t>());
        }
        else {
          std::sort(composite_accessor, composite_accessor + dim_size,
            KeyValueCompAsc<scalar_t>());
        }
      }
    }
  );
}

static void topk_kernel(
    const TensorBase &values,
    const TensorBase &indices,
    const TensorBase &self,
    int64_t k,
    int64_t dim,
    bool largest,
    bool sorted) {

  auto sizes = self.sizes();
  auto iter = TensorIteratorConfig()
    .check_all_same_dtype(false)
    .resize_outputs(false)
    .declare_static_shape(sizes, /*squash_dims=*/dim)
    .add_output(values)
    .add_output(indices)
    .add_const_input(self)
    .build();

  auto mode_values_stride = values.strides()[dim];
  auto mode_indices_stride = indices.strides()[dim];
  auto tmp_values_stride = self.strides()[dim];

  AT_DISPATCH_ALL_TYPES_AND2(ScalarType::BFloat16, ScalarType::Half, self.scalar_type(), "topk_cpu", [&] {
    auto loop = [&](char** data, const int64_t* strides, int64_t n) {
      if (self.scalar_type() == ScalarType::BFloat16) {
        return topk_impl_loop<scalar_t, float>(
            mode_values_stride, mode_indices_stride, tmp_values_stride,
            k, sizes[dim], largest, sorted, data, strides, n);
      } else {
        return topk_impl_loop<scalar_t, scalar_t>(
            mode_values_stride, mode_indices_stride, tmp_values_stride,
            k, sizes[dim], largest, sorted, data, strides, n);
      }
    };

    int64_t grain_size = internal::GRAIN_SIZE / std::max(int64_t{1}, sizes[dim]);
    iter.for_each(loop, /*grain_size=*/grain_size);
  });
}

} // anonymous namespace

ALSO_REGISTER_AVX512_DISPATCH(sort_stub, &sort_kernel);
ALSO_REGISTER_AVX512_DISPATCH(topk_stub, &topk_kernel);

} //at::native
