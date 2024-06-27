# Owner(s): ["module: cuda"]
# run time cuda tests, but with the allocator using expandable segments

import pathlib
import sys

import torch

from torch.testing._internal.common_cuda import IS_JETSON, IS_WINDOWS
from torch.testing._internal.common_utils import run_tests

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent

sys.path.insert(0, str(REPO_ROOT))
from tools.stats.import_test_stats import get_disabled_tests

from test_cuda import TestCuda, TestCudaMallocAsync, TestBlockStateAbsorption
from dynamo.test_cudagraphs import TestAotCudagraphs
from inductor.test_cudagraph_trees import CudaGraphTreeTests

if __name__ == "__main__":
    if torch.cuda.is_available() and not IS_JETSON and not IS_WINDOWS:
        get_disabled_tests(".")

        torch.cuda.memory._set_allocator_settings("expandable_segments:True")
        TestCuda.expandable_segments = lambda _: True

        run_tests()