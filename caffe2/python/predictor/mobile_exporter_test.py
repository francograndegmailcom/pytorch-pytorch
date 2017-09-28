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
from caffe2.python.test_util import TestCase
from caffe2.python import workspace, brew
from caffe2.python.model_helper import ModelHelper
from caffe2.python.predictor import mobile_exporter
import numpy as np


class TestMobileExporter(TestCase):
    def test_mobile_exporter(self):
        model = ModelHelper(name="mobile_exporter_test_model")
        # Test LeNet
        brew.conv(model, 'data', 'conv1', dim_in=1, dim_out=20, kernel=5)
        brew.max_pool(model, 'conv1', 'pool1', kernel=2, stride=2)
        brew.conv(model, 'pool1', 'conv2', dim_in=20, dim_out=50, kernel=5)
        brew.max_pool(model, 'conv2', 'pool2', kernel=2, stride=2)
        brew.fc(model, 'pool2', 'fc3', dim_in=50 * 4 * 4, dim_out=500)
        brew.relu(model, 'fc3', 'fc3')
        brew.fc(model, 'fc3', 'pred', 500, 10)
        brew.softmax(model, 'pred', 'out')

        # Create our mobile exportable networks
        workspace.RunNetOnce(model.param_init_net)
        init_net, predict_net = mobile_exporter.Export(
            workspace, model.net, model.params
        )

        # Populate the workspace with data
        np_data = np.random.rand(1, 1, 28, 28).astype(np.float32)
        workspace.FeedBlob("data", np_data)

        workspace.CreateNet(model.net)
        workspace.RunNet(model.net)
        ref_out = workspace.FetchBlob("out")

        # Clear the workspace
        workspace.ResetWorkspace()

        # Populate the workspace with data
        workspace.RunNetOnce(init_net)
        # Fake "data" is populated by init_net, we have to replace it
        workspace.FeedBlob("data", np_data)

        # Overwrite the old net
        workspace.CreateNet(predict_net, True)
        workspace.RunNet(predict_net.name)
        manual_run_out = workspace.FetchBlob("out")
        np.testing.assert_allclose(
            ref_out, manual_run_out, atol=1e-10, rtol=1e-10
        )

        # Clear the workspace
        workspace.ResetWorkspace()

        # Predictor interface test (simulates writing to disk)
        predictor = workspace.Predictor(
            init_net.SerializeToString(), predict_net.SerializeToString()
        )

        # Output is a vector of outputs but we only care about the first and only result
        predictor_out = predictor.run([np_data])
        assert len(predictor_out) == 1
        predictor_out = predictor_out[0]

        np.testing.assert_allclose(
            ref_out, predictor_out, atol=1e-10, rtol=1e-10
        )
