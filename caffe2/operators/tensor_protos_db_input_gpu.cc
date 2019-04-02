#include "caffe2/core/common_gpu.h"
#include "caffe2/core/context_gpu.h"
#include "caffe2/operators/tensor_protos_db_input.h"

namespace caffe2 {
namespace {
REGISTER_CUDA_OPERATOR(TensorProtosDBInput, TensorProtosDBInput<CUDAContext>);
}  // namespace
}  // namespace caffe2
