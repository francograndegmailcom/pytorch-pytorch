#include <c10d/NCCLUtils.hpp>

#ifdef USE_C10D_NCCL

#include <mutex>

namespace c10d {

std::string getNcclVersion() {
  static std::once_flag ncclGetVersionFlag;
  static std::string versionString;

  std::call_once(ncclGetVersionFlag, []() {
    int version;
    ncclResult_t status = ncclGetVersion(&version);
    // can't compute the version if call did not return successfully or version
    // code < 100 (corresponding to 0.1.0)
    if (status != ncclSuccess || version < 100) {
      versionString = "Unknown NCCL version";
    } else {
      // logic reference: https://github.com/NVIDIA/nccl/blob\
      // /7e515921295adaab72adf56ea71a0fafb0ecb5f3/src/nccl.h.in#L22
      int majorDiv = (version >= 10000)? 10000 : 1000;
      auto ncclMajor = version / majorDiv;
      auto ncclMinor = (version % majorDiv) / 100;
      auto ncclPatch = version % (ncclMajor * majorDiv + ncclMinor * 100);
      versionString = std::to_string(ncclMajor) + "." +
          std::to_string(ncclMinor) + "." + std::to_string(ncclPatch);
    }
  });

  return versionString;
}

std::string ncclGetErrorWithVersion(ncclResult_t error) {
  return std::string(ncclGetErrorString(error)) + ", NCCL version " +
      getNcclVersion();
}

} // namespace c10d

#endif // USE_C10D_NCCL
