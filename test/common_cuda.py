r"""This file may initialize CUDA context when imported."""

import torch
import torch.cuda


TEST_CUDA = torch.cuda.is_available()
TEST_MULTIGPU = TEST_CUDA and torch.cuda.device_count() >= 2
CUDA_DEVICE = TEST_CUDA and torch.device("cuda:0")
TEST_CUDNN = TEST_CUDA and torch.backends.cudnn.is_acceptable(torch.tensor(1., device=CUDA_DEVICE))
TEST_CUDNN_VERSION = TEST_CUDNN and torch.backends.cudnn.version()


# Used below in `initialize_cuda_context_rng` to ensure that CUDA context and
# RNG have been initialized.
__cuda_ctx_rng_initialized = False


# after this call, CUDA context and RNG must have been initialized on each GPU
def initialize_cuda_context_rng():
    global __cuda_ctx_rng_initialized
    assert TEST_CUDA, 'CUDA must be available when calling initialize_cuda_context_rng'
    if not __cuda_ctx_rng_initialized:
        # initialize cuda context and rng for memory tests
        for i in range(torch.cuda.device_count()):
            torch.randn(1, device="cuda:{}".format(i))
        __cuda_ctx_rng_initialized = True
