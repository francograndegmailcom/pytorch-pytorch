#include <ATen/ATen.h>
#include <ATen/NativeFunctions.h>
#include <ATen/LegacyTHFunctions.h>

namespace at {
namespace native {

Tensor& upsample_nearest1d_out_cuda(
    Tensor& output,
    const Tensor& input,
    IntArrayRef output_size) {
    return at::legacy::th::_thnn_upsample_nearest1d_forward_out(
        output, input, output_size);
}

Tensor upsample_nearest1d_cuda(
    const Tensor& input,
    IntArrayRef output_size) {
    auto output = at::empty({0}, input.options());
    return at::legacy::th::_thnn_upsample_nearest1d_forward(
        output, input, output_size);
}

Tensor& upsample_nearest1d_backward_out_cuda(
    Tensor& grad_input,
    const Tensor& grad_output,
    IntArrayRef output_size,
    IntArrayRef input_size) {
    return at::legacy::th::_thnn_upsample_nearest1d_backward_out(
        grad_input, grad_output, output_size, input_size);
}

Tensor upsample_nearest1d_backward_cuda(
    const Tensor& grad_output,
    IntArrayRef output_size,
    IntArrayRef input_size) {
    auto grad_input = at::zeros_like(grad_output);
    return at::legacy::th::_thnn_upsample_nearest1d_backward(
        grad_input, grad_output, output_size, input_size);
}

namespace {

} // namespace

} // native
} // at