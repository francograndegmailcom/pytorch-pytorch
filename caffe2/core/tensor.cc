#include "caffe2/core/tensor.h"

#include "caffe2/core/blob_stats.h"

namespace caffe2 {

CAFFE_DEFINE_PREALLOCATED_KNOWN_TYPE(12, Tensor);

TensorPrinter::TensorPrinter(
    const std::string& tensor_name,
    const std::string& file_name,
    int limit)
    : to_file_(!file_name.empty()),
      limit_(limit ? limit : k_limit_default_),
      tensor_name_(tensor_name) {
  if (to_file_) {
    // We will output to file instead of printing on screen.
    // We will write each individual tensor to its individual file.
    log_file_.reset(new std::ofstream(
        file_name, std::ofstream::out | std::ofstream::trunc));
    CAFFE_ENFORCE(
        log_file_->good(),
        "Failed to open TensorPrinter file ",
        file_name,
        ". rdstate() = ",
        log_file_->rdstate());
  }
}

TensorPrinter::~TensorPrinter() {
  if (log_file_.get()) {
    log_file_->close();
  }
}

void TensorPrinter::PrintMeta(const Tensor& tensor) {
  if (to_file_) {
    (*log_file_) << MetaStr(tensor) << std::endl;
  } else {
    LOG(INFO) << MetaStr(tensor);
  }
}

std::string TensorPrinter::MetaStr(const Tensor& tensor) {
  std::stringstream meta_stream;
  meta_stream << "Tensor " << tensor_name_ << " of type "
              << tensor.meta().name() << ". Dims: (";
  for (const auto dim : tensor.sizes()) {
    meta_stream << dim << ",";
  }
  meta_stream << "): ";
  return meta_stream.str();
}

TypeMeta GetTensorType(const void* c) {
  const Tensor* tc = static_cast<const Tensor*>(c);
  return tc->meta();
}

// TODO(jerryzh): Remove
static CaffeMap<TypeIdentifier, TypeCall> type_call_registry_{
    {TypeMeta::Id<Tensor>(), GetTensorType}};

TypeCall GetTypeCallFunction(TypeIdentifier id) {
  auto f = type_call_registry_.find(id);
  if (f == type_call_registry_.end()) {
    return nullptr;
  }
  return f->second;
}

void RegisterTypeCallFunction(TypeIdentifier id, TypeCall c) {
  type_call_registry_[id] = c;
}

int GetGPUIDForPointer(const void* ptr);

vector<int64_t> GetTensorInfo(
    const void* c,
    size_t* capacity,
    DeviceOption* device) {
  CHECK(capacity);
  const Tensor* tc = static_cast<const Tensor*>(c);
  CHECK(tc);
  CHECK(tc->unsafeGetTensorImpl());
  CHECK(tc->unsafeGetTensorImpl()->storage().unsafeGetStorageImpl());
  *capacity = tc->storage().capacity();
  ExtractDeviceOption(device, tc->GetDevice());
  return tc->sizes().vec();
}

// since we only have one tensor, probably need to remove this at some point?
static CaffeMap<TypeIdentifier, TensorInfoCall> tensor_info_call_registry_{
    {TypeMeta::Id<Tensor>(), GetTensorInfo}};

// TODO: Remove this code in a separate diff, since we only have one
// GetTensorInfo function now
TensorInfoCall GetTensorInfoFunction(TypeIdentifier id) {
  auto f = tensor_info_call_registry_.find(id);
  if (f == tensor_info_call_registry_.end()) {
    return nullptr;
  }
  return f->second;
}

void RegisterTensorInfoFunction(TypeIdentifier id, TensorInfoCall c) {
  tensor_info_call_registry_[id] = c;
}

void TensorVectorResize(
    std::vector<Tensor>& tensors,
    int size,
    DeviceType type) {
  tensors.reserve(size);
  for (auto i = 0; i < size; ++i) {
    tensors.emplace_back(type);
  }
}

Tensor empty(at::IntList dims, at::TensorOptions options) {
  // TODO: merge this with at::empty after Tensor is merged
  auto tensor = Tensor(dims, options.device().type());
  tensor.raw_mutable_data(scalarTypeToTypeMeta(options.dtype()));
  return tensor;
}

namespace {

struct TensorStatGetter : BlobStatGetter {
  size_t sizeBytes(const Blob& blob) const override {
    const auto& tensor = blob.Get<Tensor>();
    auto nbytes = tensor.nbytes();
    if (nbytes > 0 && tensor.IsType<std::string>()) {
      const auto* data = tensor.data<std::string>();
      for (int i = 0; i < tensor.size(); ++i) {
        nbytes += data[i].size();
      }
    }
    return nbytes;
  }
};
REGISTER_BLOB_STAT_GETTER(Tensor, TensorStatGetter);
}

} // namespace caffe2
