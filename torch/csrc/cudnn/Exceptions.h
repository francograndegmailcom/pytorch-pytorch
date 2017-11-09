#pragma once

#include "Types.h"

#include "cudnn-wrapper.h"
#include <string>
#include <stdexcept>
#include <sstream>

#include <THC/THC.h>

#define CHECK_ARG(cond) _CHECK_ARG(cond, #cond, __FILE__, __LINE__)

extern THCState* state; // TO DELETE

namespace torch { namespace cudnn {

inline int assertSameGPU(const at::Tensor& t) {
  return t.get_device();
}

template<typename ...T>
int assertSameGPU(const at::Tensor& t, T& ... tensors) {
  static_assert(std::is_same<at::Tensor, typename std::common_type<T...>::type>::value,
      "all arguments to assertSameGPU have to be at::Tensor&");
  auto t_device = t.get_device();
  if (t_device != assertSameGPU(tensors...))
    throw std::runtime_error("tensors are on different GPUs");
  return t_device;
}

class cudnn_exception : public std::runtime_error {
public:
  cudnnStatus_t status;
  cudnn_exception(cudnnStatus_t status, const char* msg)
      : std::runtime_error(msg)
      , status(status) {}
  cudnn_exception(cudnnStatus_t status, const std::string& msg)
      : std::runtime_error(msg)
      , status(status) {}
};

inline void CHECK(cudnnStatus_t status)
{
  if (status != CUDNN_STATUS_SUCCESS) {
    if (status == CUDNN_STATUS_NOT_SUPPORTED) {
        throw cudnn_exception(status, std::string(cudnnGetErrorString(status)) +
                ". This error may appear if you passed in a non-contiguous input.");
    }
    throw cudnn_exception(status, cudnnGetErrorString(status));
  }
}

inline void CUDA_CHECK(cudaError_t error)
{
  if (error) {
    std::string msg("CUDA error: ");
    msg += cudaGetErrorString(error);
    throw std::runtime_error(msg);
  }
}

inline void _CHECK_ARG(bool cond, const char* code, const char* f, int line) {
  if (!cond) {
    std::stringstream ss;
    ss << "CHECK_ARG(" << code << ") failed at " << f << ":" << line;
    throw std::runtime_error(ss.str());
  }
}

}}  // namespace torch::cudnn
