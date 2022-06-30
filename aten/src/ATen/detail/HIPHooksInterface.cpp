#include <ATen/detail/HIPHooksInterface.h>

#include <c10/util/Exception.h>
#include <c10/util/CallOnce.h>

#include <cstddef>
#include <memory>
#include <mutex>

namespace at {
namespace detail {

// See getCUDAHooks for some more commentary
const HIPHooksInterface& getHIPHooks() {
  static std::unique_ptr<HIPHooksInterface> hip_hooks;
#if !defined C10_MOBILE
  static c10::once_flag once;
  c10::call_once(once, [] {
    hip_hooks = HIPHooksRegistry()->Create("HIPHooks", HIPHooksArgs{});
    if (!hip_hooks) {
      hip_hooks =
          // NOLINTNEXTLINE(modernize-make-unique)
          std::unique_ptr<HIPHooksInterface>(new HIPHooksInterface());
    }
  });
#else
  if (hip_hooks == nullptr) {
    hip_hooks =
        // NOLINTNEXTLINE(modernize-make-unique)
        std::unique_ptr<HIPHooksInterface>(new HIPHooksInterface());
  }
#endif
  return *hip_hooks;
}
} // namespace detail

C10_DEFINE_REGISTRY(HIPHooksRegistry, HIPHooksInterface, HIPHooksArgs)

} // namespace at
