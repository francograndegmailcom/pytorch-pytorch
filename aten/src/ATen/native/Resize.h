#pragma once

#include "ATen/ATen.h"
#include "TH/THTensor.hpp"

namespace at { namespace native {

// These functions are called by native::resize_ as well as (legacy) TH resize.
// They are not in TH/THTensor.cpp because the at namespace is easier
// to benchmark than TH; I can't get gbenchmark to call fns from THTensor.cpp

static inline int64_t computeStorageSize(
    IntList sizes, IntList strides, int64_t storage_offset) {
  int64_t result = 1;
  for (size_t dim = 0; dim < sizes.size(); ++dim) {
    // NB: (a tensor with arbitrary 0 dim)'s storage_size is 0.
    if (sizes[dim] == 0) {
      return 0;
    }
    result += strides[dim] * (sizes[dim] - 1);
  }
  return result + storage_offset;
}

static inline void maybe_resize_storage_cpu(TensorImpl* self, int64_t new_size) {
  if (new_size == 0) {
    return;
  }
  if (!THTensor_getStoragePtr(self)) {
    THTensor_stealAndSetStoragePtr(self, THStorage_new(self->dtype()));
  }
  if (new_size > self->storage().numel()) {
    THStorage_resize(THTensor_getStoragePtr(self), new_size);
  }
}

inline TensorImpl* resize_impl_cpu_(
    TensorImpl* self,
    IntList size,
    c10::optional<IntList> stride) {
  if (self->sizes() == size && (!stride || self->strides() == stride)) {
    return self;
  }

  int64_t storage_offset = self->storage_offset();
  int64_t storage_size = 1;
  if (stride) {
    self->set_sizes_and_strides(size, *stride);
    storage_size = computeStorageSize(size, *stride, storage_offset);
  } else {
    self->set_sizes_contiguous(size);
    storage_size = self->numel() + storage_offset;
  }
  maybe_resize_storage_cpu(self, storage_size);

  return self;
}

static inline void checkInBoundsForStorage(
    IntList size,
    IntList stride,
    int64_t storage_offset,
    const Storage& new_storage) {
  int64_t storage_size = computeStorageSize(size, stride, storage_offset);
  if (storage_size == 0) {
    // NB: (a tensor with arbitrary 0 dims)'s storage can have any numel.
    return;
  }
  int64_t new_storage_size = new_storage.numel();
  AT_CHECK(
      storage_size <= new_storage_size,
      "setStorage: sizes ", size, ", strides ", stride, ","
      " and storage offset ", storage_offset,
      " requiring a storage size of ", storage_size,
      " are out of bounds for storage with numel ", new_storage_size);
}

/**
 * Set self's sizes, strides, and storage_offset.
 * (size, stride, storage_offset) must be in bounds for self's storage.
 */
inline void setStrided(
    const Tensor& self,
    IntList size,
    IntList stride,
    int64_t storage_offset) {
  auto* self_ = self.unsafeGetTensorImpl();
  checkInBoundsForStorage(size, stride, storage_offset, self_->storage());

  /* storage offset */
  AT_CHECK(storage_offset >= 0, "Tensor: invalid storage offset ", storage_offset);
  self_->set_storage_offset(storage_offset);

  /* size and stride */
  AT_ASSERT(size.size() == stride.size());
  if (self_->sizes() == size && self_->strides() == stride) {
    return;
  }
  self_->set_sizes_and_strides(size, stride);
}

}}
