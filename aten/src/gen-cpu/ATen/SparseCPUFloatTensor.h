#pragma once

#include <TH/TH.h>
#include <THNN/THNN.h>
#undef THNN_
#include <THS/THS.h>

#include "ATen/Tensor.h"
#include "ATen/TensorImpl.h"
#include "ATen/Context.h"
#include "ATen/TensorMethods.h"

namespace at {

struct SparseCPUFloatTensor final : public TensorImpl {
public:
  explicit SparseCPUFloatTensor(Context* context);
  SparseCPUFloatTensor(Context* context, THSFloatTensor * tensor);
  virtual ~SparseCPUFloatTensor();
  virtual const char * toString() const override;
  virtual IntList sizes() const override;
  virtual IntList strides() const override;
  virtual int64_t dim() const override;
  virtual Scalar localScalar() override;
  virtual void * unsafeGetTH(bool retain) override;
  virtual std::unique_ptr<Storage> storage() override;
  static const char * typeString();

//TODO(zach): sort of friend permissions later so this
// can be protected
public:
  THSFloatTensor * tensor;
  Context* context;
  friend struct SparseCPUFloatType;
};

} // namespace at
