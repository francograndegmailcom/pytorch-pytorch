#pragma once

#include <string>
#include <sstream>
#include <unordered_map>
#include <vector>

#include <c10/util/irange.h>
#include <ATen/cuda/detail/LazyNVRTC.h>

#ifdef __HIP_PLATFORM_HCC__
    // pass -- jiterator not supported on HIP platforms
    #define jiterator_stringify(...) std::string("USE_JITERATOR is undefined");
#else
    #define jiterator_stringify(...) std::string(#__VA_ARGS__);
#endif

namespace at { namespace cuda { namespace jit {

enum class BinaryFuncVariant {NoScalar, RhsScalar, LhsScalar};

struct NvrtcFunction {
  CUmodule module = CUmodule();
  CUfunction function = nullptr;
};

std::string generate_code(
    int nTensors,
    const std::string& func,
    const std::string& name,
    const std::string& f_input_type,
    const std::string& compute_type,
    const std::string& result_type,
    bool contiguous,
    bool dynamic_casting,
    BinaryFuncVariant scalar_pos,
    c10::SmallVector<std::string>& extra_args_typenames,
    bool vectorized=false,
    int vec_size=0);

NvrtcFunction jit_pwise_function(
    const std::string& code,
    const std::string& kernel_name);

void launch_jitted_pwise_function(
    NvrtcFunction function,
    void* args[],
    const int nBlocks,
    const int kBlockSize);

template <typename T>
struct delayed_false : std::false_type {
};

// Defines type names
// NOTE: General case is instantiated only for invalid types.
// All the valid types have specialization using the TYPE_NAME_FN
// macro below.
template <typename T>
inline std::string typeName() {
  // we can't use static_assert(false) directly as the
  // program will be not compile even if the template is not
  // instantiated, so we use `delayed_false`
  // to make sure compiler doesn't eagerly raise
  // fail this assertion.
  static_assert(delayed_false<T>::value, "invalid type for jiterator");
  return "void";
}

#define TYPE_NAME_FN(ctype, name) \
template <> inline std::string typeName<ctype>(){ \
    return std::string(#ctype);    \
}

AT_FORALL_SCALAR_TYPES_WITH_COMPLEX_AND_QINTS(TYPE_NAME_FN)

}}}  // namespace at::cuda::jit
