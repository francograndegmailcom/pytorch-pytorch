#include <c10/util/Exception.h>
#include <torch/csrc/api/include/torch/serialize/istream_adapter.h>

namespace torch::serialize {

size_t IStreamAdapter::size() const {
  auto prev_pos = istream_->tellg();
  validate("getting the current position");
  istream_->seekg(0, istream_->end);
  validate("seeking to end");
  auto result = istream_->tellg();
  validate("getting size");
  istream_->seekg(prev_pos);
  validate("seeking to the original position");
  return result;
}

size_t IStreamAdapter::read(uint64_t pos, void* buf, size_t n, const char* what)
    const {
  istream_->seekg(pos);
  validate(what);
  istream_->read(static_cast<char*>(buf), n);
  validate(what);
  return n;
}

void IStreamAdapter::validate(const char* what) const {
  if (!*istream_) {
    AT_ERROR("istream reader failed: ", what, ".");
  }
}

} // namespace torch::serialize
