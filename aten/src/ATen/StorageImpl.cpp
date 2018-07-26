#include <ATen/Context.h>
#include <ATen/StorageImpl.h>

namespace at {

StorageImpl::StorageImpl(
    at::ScalarType scalar_type,
    ptrdiff_t size,
    at::DataPtr data_ptr,
    at::Allocator* allocator,
    bool resizeable)
    : scalar_type(scalar_type),
      data_ptr(std::move(data_ptr)),
      size(size),
      refcount(1),
      weakcount(1), // from the strong reference
      resizeable(resizeable),
      allocator(allocator),
      finalizer(nullptr) {}

StorageImpl::StorageImpl(
    at::ScalarType scalar_type,
    ptrdiff_t size,
    at::Allocator* allocator,
    bool resizeable)
    : StorageImpl(
          scalar_type,
          size,
          allocator->allocate(at::elementSize(scalar_type) * size),
          allocator,
          resizeable) {}

Type& StorageImpl::type() {
  if (data_ptr.device().is_cuda()) {
    return globalContext().getType(Backend::CUDA, scalar_type);
  }
  return globalContext().getType(Backend::CPU, scalar_type);
}

} // namespace at
