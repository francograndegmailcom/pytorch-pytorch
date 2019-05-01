#include <ATen/native/mkldnn/MKLDNNCommon.h>
#include <ATen/native/utils/ParamUtils.h>
#include <ATen/native/mkldnn/MKLDNNConversions.h>

namespace at { namespace native {

#if AT_MKLDNN_ENABLED()

Tensor mkldnn_to_dense(const Tensor& mkldnn_tensor) {
  ideep::tensor& stensor = itensor_from_mkldnn(mkldnn_tensor);
  auto dims = stensor.get_dims();
  // NOTE: int32_t dims from ideep::tensor but sizes needs int64_t
  Tensor cpu_tensor = at::empty(
    std::vector<int64_t>(dims.begin(), dims.end()),
    mkldnn_tensor.options().layout(c10::kStrided));
  stensor.reorder_to(cpu_tensor.template data<float>());
  return cpu_tensor;
}

Tensor dense_to_mkldnn(const Tensor& cpu_tensor) {
  AT_ASSERTM(cpu_tensor.type_id() == CPUTensorId(),
             "dense_to_mkldnn expects dense CPU tensor input");
  AT_ASSERTM(cpu_tensor.scalar_type() == ScalarType::Float,
             "dense_to_mkldnn expects float tensor input");
  // TODO: consider to convert non-contiguous tensor to `ideep::tensor` directly.
  auto cpu_tensor_cont = cpu_tensor.contiguous();
  Tensor mkldnn_tensor = new_with_sizes_mkldnn(cpu_tensor_cont.sizes(), cpu_tensor_cont.options());
  ideep::tensor& dtensor = itensor_from_mkldnn(mkldnn_tensor);
  dtensor.reorder_from(dtensor.get_dims(),
                       ideep::tensor::data_type::f32,
                    (cpu_tensor_cont.template data<float>()));
  return mkldnn_tensor;
}

Tensor mkldnn_reorder_conv2d_input(
    const Tensor& self,
    const Tensor& weight,
    IntArrayRef padding,
    IntArrayRef stride,
    IntArrayRef dilation,
    int64_t groups) {
  AT_ASSERTM(self.type_id() == CPUTensorId(),
             "dense_to_mkldnn expects dense CPU tensor input");
  AT_ASSERTM(self.scalar_type() == ScalarType::Float,
             "mkldnn_reorder_conv2d_input expects float tensor input");

  auto stride_vec = expand_param_if_needed(stride, "stride", 2);
  auto padding_vec = expand_param_if_needed(padding, "padding", 2);
  auto dilation_vec = expand_param_if_needed(dilation, "dilation", 2);

  ideep::tensor::descriptor desc =
    ideep::convolution_forward::expected_src_descriptor(
        {weight.sizes().cbegin(), weight.sizes().cend()},
        ideep::tensor::data_type::f32,
        {stride_vec.cbegin(), stride_vec.cend()},
        {padding_vec.cbegin(), padding_vec.cend()},
        {padding_vec.cbegin(), padding_vec.cend()},
        {dilation_vec.cbegin(), dilation_vec.cend()},
        groups,
        ideep::algorithm::convolution_direct,
        ideep::prop_kind::forward,
        ideep::tensor::data_type::f32,
        {self.sizes().cbegin(), self.sizes().cend()});
  ideep::tensor result(desc);

  result.reorder_from(
      result.get_dims(),
      result.get_data_type(),
      self.template data<float>());

  return new_with_itensor_mkldnn(std::move(result), self.options());
}

Tensor mkldnn_reorder_conv2d_weight(
    const Tensor& self,
    IntArrayRef padding,
    IntArrayRef stride,
    IntArrayRef dilation,
    int64_t groups) {
  AT_ASSERTM(self.scalar_type() == ScalarType::Float,
             "mkldnn_reorder_conv2d_weight expects float tensor input");

  auto stride_vec = expand_param_if_needed(stride, "stride", 2);
  auto padding_vec = expand_param_if_needed(padding, "padding", 2);
  auto dilation_vec = expand_param_if_needed(dilation, "dilation", 2);

  ideep::tensor::descriptor desc =
    ideep::convolution_forward::expected_weights_descriptor(
        {self.sizes().cbegin(), self.sizes().cend()},
        ideep::tensor::data_type::f32,
        {stride_vec.cbegin(), stride_vec.cend()},
        {padding_vec.cbegin(), padding_vec.cend()},
        {padding_vec.cbegin(), padding_vec.cend()},
        {dilation_vec.cbegin(), dilation_vec.cend()},
        groups,
        ideep::algorithm::convolution_direct,
        ideep::prop_kind::forward);
  ideep::tensor result(desc);

  if (self.is_mkldnn()) {
    ideep::tensor w = itensor_from_mkldnn(self).as_weights();
    w.make_group(groups);
    ideep::reorder::compute(w, result);
  } else {
    result.reorder_from(
        result.get_dims(),
        result.get_data_type(),
        self.template data<float>());
  }
  return new_with_itensor_mkldnn(std::move(result), self.options());
}

#else

Tensor mkldnn_to_dense(const Tensor& mkldnn_tensor) {
  AT_ERROR("MKL-DNN build is disabled");
}

Tensor dense_to_mkldnn(const Tensor& cpu_tensor) {
  AT_ERROR("MKL-DNN build is disabled");
}

Tensor mkldnn_reorder_conv2d_weight(
    const Tensor& self,
    IntArrayRef padding,
    IntArrayRef stride,
    IntArrayRef dilation,
    int64_t groups) {
  AT_ERROR("mkldnn_reorder_conv2d_weight: MKL-DNN build is disabled");
}

#endif // AT_MKLDNN_ENABLED()

}}
