# Copyright (c) 2016-present, Facebook, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
##############################################################################

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import unittest
import hypothesis.strategies as st
from hypothesis import given, settings
import numpy as np
from caffe2.python import core, workspace
import caffe2.python.hypothesis_test_util as hu
import caffe2.python.mkl_test_util as mu


@unittest.skipIf(not workspace.C.has_mkldnn,
                 "Skipping as we do not have mkldnn.")
class MKLConvTest(hu.HypothesisTestCase):
    @given(stride=st.integers(1, 3),
           pad=st.integers(0, 3),
           kernel=st.integers(3, 5),
           size=st.integers(8, 8),
           input_channels=st.integers(1, 3),
           output_channels=st.integers(1, 3),
           batch_size=st.integers(1, 3),
           **mu.gcs)
    @settings(max_examples=2, timeout=100)
    def test_mkl_convolution(self, stride, pad, kernel, size,
                             input_channels, output_channels,
                             batch_size, gc, dc):
        op = core.CreateOperator(
            "Conv",
            ["X", "w", "b"],
            ["Y"],
            stride=stride,
            pad=pad,
            kernel=kernel,
        )
        X = np.random.rand(
            batch_size, input_channels, size, size).astype(np.float32) - 0.5
        w = np.random.rand(
                output_channels, input_channels, kernel, kernel) \
            .astype(np.float32) - 0.5
        b = np.random.rand(output_channels).astype(np.float32) - 0.5

        inputs = [X, w, b]
        self.assertDeviceChecks(dc, op, inputs, [0])


if __name__ == "__main__":
    import unittest
    unittest.main()
