#include <torch/optim/optimizer.h>

#include <torch/nn/cursor.h>
#include <torch/serialize/base.h>
#include <torch/tensor.h>

#include <string>
#include <utility>
#include <vector>

namespace torch {
namespace optim {
namespace detail {
void serialize(
    serialize::Writer& writer,
    const std::string& key,
    const std::vector<int64_t>& steps) {
  std::vector<torch::Tensor> tensors;
  for (const auto& step : steps) {
    tensors.push_back(torch::tensor(static_cast<int64_t>(step)));
  }
  writer.write(key, tensors, /*is_buffer=*/true);
}

void serialize(
    serialize::Reader& reader,
    const std::string& key,
    std::vector<int64_t>& steps) {
  std::vector<torch::Tensor> tensors;
  reader.read(key, tensors, /*is_buffer=*/true);
  steps.clear();
  for (const auto& step : tensors) {
    steps.push_back(step.toCLong());
  }
}

OptimizerBase::OptimizerBase(std::vector<Tensor> parameters)
    : parameters_(std::move(parameters)) {}

OptimizerBase::OptimizerBase(const ParameterCursor& cursor) {
  add_parameters(cursor);
}

void OptimizerBase::add_parameters(const std::vector<Tensor>& parameters) {
  parameters_.insert(parameters_.end(), parameters.begin(), parameters.end());
}

void OptimizerBase::add_parameters(const ParameterCursor& cursor) {
  std::vector<Tensor> tensors(cursor.size());
  cursor.map(tensors.begin(), [](const Tensor& tensor) { return tensor; });
  add_parameters(tensors);
}

void OptimizerBase::zero_grad() {
  for (auto& parameter : parameters_) {
    if (parameter.grad().defined()) {
      parameter.grad().detach_();
      parameter.grad().zero_();
    }
  }
}

const std::vector<Tensor>& OptimizerBase::parameters() const noexcept {
  return parameters_;
}

std::vector<Tensor>& OptimizerBase::parameters() noexcept {
  return parameters_;
}

size_t OptimizerBase::size() const noexcept {
  return parameters_.size();
}
} // namespace detail
} // namespace optim
} // namespace torch
