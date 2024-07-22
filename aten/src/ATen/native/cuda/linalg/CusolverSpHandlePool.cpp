#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/detail/DeviceThreadHandles.h>

#if defined(CUDART_VERSION) || defined(USE_ROCM)

namespace at::cuda {
namespace {

void createCusolverSpHandle(cusolverSpHandle_t *handle) {
  TORCH_CUSOLVER_CHECK(cusolverSpCreate(handle));
}

void destroyCusolverSpHandle(cusolverSpHandle_t handle) {
// this is because of something dumb in the ordering of
// destruction. Sometimes atexit, the cuda context (or something)
// would already be destroyed by the time this gets destroyed. It
// happens in fbcode setting. @colesbury and @soumith decided to not destroy
// the handle as a workaround.
//   - Comments of @soumith copied from cuDNN handle pool implementation
#ifdef NO_CUDNN_DESTROY_HANDLE
  (void)handle; // Suppress unused variable warning
#else
    cusolverSpDestroy(handle);
#endif
}

using CuSolverSpPoolType = DeviceThreadHandlePool<cusolverSpHandle_t, createCusolverSpHandle, destroyCusolverSpHandle>;

} // namespace

cusolverSpHandle_t getCurrentCUDASolverSpHandle() {
  c10::DeviceIndex device = 0;
  AT_CUDA_CHECK(c10::cuda::GetDevice(&device));

  // Thread local PoolWindows are lazily-initialized
  // to avoid initialization issues that caused hangs on Windows.
  // See: https://github.com/pytorch/pytorch/pull/22405
  // This thread local unique_ptrs will be destroyed when the thread terminates,
  // releasing its reserved handles back to the pool.
  static auto pool = std::make_shared<CuSolverSpPoolType>();
  thread_local std::unique_ptr<CuSolverSpPoolType::PoolWindow> myPoolWindow(
      pool->newPoolWindow());

  auto handle = myPoolWindow->reserve(device);
  auto stream = c10::cuda::getCurrentCUDAStream();
  TORCH_CUSOLVER_CHECK(cusolverSpSetStream(handle, stream));
  return handle;
}

} // namespace at::cuda

#endif // CUDART_VERSION
