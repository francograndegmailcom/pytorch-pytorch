#ifndef CAFFE2_OPERATORS_REVERSE_PACKED_SEGS_OP_H_
#define CAFFE2_OPERATORS_REVERSE_PACKED_SEGS_OP_H_

#include "caffe2/core/context.h"
#include "caffe2/core/operator.h"

namespace caffe2 {

template <class Context>
class ReversePackedSegsOp final : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  USE_SIMPLE_CTOR_DTOR(ReversePackedSegsOp);
  USE_DISPATCH_HELPER;

  bool RunOnDevice() override {
    return DispatchHelper<TensorTypes<float, double, int, long, bool>>::call(
        this, Input(DATA));
  }

  template <typename T>
  bool DoRunWithType() {
    if (OperatorBase::Input<Tensor<CPUContext>>(LENGTHS)
            .template IsType<int>()) {
      DoRunWithLengthType<T, int>();
    } else {
      DoRunWithLengthType<T, long>();
    }
    return true;
  }

 private:
  INPUT_TAGS(DATA, LENGTHS);

  template <typename T, typename LengthType>
  void DoRunWithLengthType() {
    const auto& data = Input(DATA);
    const auto& lengths = OperatorBase::Input<Tensor<CPUContext>>(LENGTHS);

    CAFFE_ENFORCE(
        data.ndim() == 3,
        "DATA should be 3-D tensor <lengths, "
        "segments, embeddings>");
    CAFFE_ENFORCE(lengths.ndim() == 1, "LENGTH should be 1-D");

    auto* output = Output(0);
    const auto& shape = data.dims();
    output->Resize(shape);

    const auto& max_length = data.dims()[0];
    const auto& batch_size = data.dims()[1];
    const auto& block_size = data.dims()[2];
    CAFFE_ENFORCE(
        lengths.dims()[0] == batch_size,
        "lenths size should be"
        " equal to batch size");

    const T* data_ptr = data.template data<T>();
    const LengthType* lengths_ptr = lengths.template data<LengthType>();

    T* rev_data_ptr = output->template mutable_data<T>();
    for (TIndex i = 0; i < batch_size; i++) {
      const auto& seg_length = lengths_ptr[i];
      CAFFE_ENFORCE_LE(seg_length, max_length);
      TIndex j = 0;
      for (; j < seg_length; j++) {
        const T* data_block_ptr = data_ptr + (j * batch_size + i) * block_size;
        T* rev_data_block_ptr =
            rev_data_ptr + ((seg_length - 1 - j) * batch_size + i) * block_size;
        context_.template Copy<T, Context, Context>(
            block_size, data_block_ptr, rev_data_block_ptr);
      }
      for (; j < max_length; j++) {
        const T* data_block_ptr = data_ptr + (j * batch_size + i) * block_size;
        T* rev_data_block_ptr =
            rev_data_ptr + (j * batch_size + i) * block_size;
        context_.template Copy<T, Context, Context>(
            block_size, data_block_ptr, rev_data_block_ptr);
      }
    }
  }
};

} // namespace caffe2

#endif // CAFFE2_OPERATORS_REVERSE_PACKED_SEGS_OP_H_
