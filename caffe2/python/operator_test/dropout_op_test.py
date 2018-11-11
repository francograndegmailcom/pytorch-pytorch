from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from hypothesis import assume, given
import hypothesis.strategies as st
import numpy as np

from caffe2.proto import caffe2_pb2
from caffe2.python import core, workspace
import caffe2.python.hypothesis_test_util as hu
import caffe2.python.serialized_test.serialized_test_util as serial


class TestDropout(serial.SerializedTestCase):
    @serial.given(
        X=hu.tensor(),
        in_place=st.booleans(),
        ratio=st.floats(0, 0.999),
        engine=st.sampled_from(["", "CUDNN"]),
        **hu.gcs)
    def test_dropout_is_test(self, X, in_place, ratio, engine, gc, dc):
        """Test with is_test=True for a deterministic reference impl."""
        op = core.CreateOperator(
            "Dropout", ["X"], ["X" if in_place else "Y"],
            ratio=ratio,
            engine=engine,
            is_test=True)

        self.assertDeviceChecks(dc, op, [X], [0])

        # No sense in checking gradients for test phase

        def reference_dropout_test(x):
            return x, np.ones(x.shape, dtype=np.bool)

        self.assertReferenceChecks(
            gc,
            op,
            [X],
            reference_dropout_test,
            # The 'mask' output may be uninitialized
            outputs_to_check=[0])

    @given(
        X=hu.tensor(),
        in_place=st.booleans(),
        output_mask=st.booleans(),
        engine=st.sampled_from(["", "CUDNN"]),
        **hu.gcs)
    def test_dropout_ratio0(self, X, in_place, output_mask, engine, gc, dc):
        """Test with ratio=0 for a deterministic reference impl."""
        is_test = not output_mask
        op = core.CreateOperator(
            "Dropout", ["X"],
            ["X" if in_place else "Y"] + (["mask"] if output_mask else []),
            ratio=0.0,
            engine=engine,
            is_test=is_test)

        self.assertDeviceChecks(dc, op, [X], [0])
        if not is_test:
            self.assertGradientChecks(gc, op, [X], 0, [0])

        def reference_dropout_ratio0(x):
            return (x, ) if is_test else (x, np.ones(x.shape, dtype=np.bool))

        self.assertReferenceChecks(
            gc,
            op,
            [X],
            reference_dropout_ratio0,
            # Don't check the mask with cuDNN because it's packed data
            outputs_to_check=None if engine != 'CUDNN' else [0])

    @given(
        N=st.integers(1000, 2000),
        ratio=st.floats(0.2, 0.8),
        in_place=st.booleans(),
        engine=st.sampled_from(["", "CUDNN"]),
        **hu.gcs)
    def test_dropout_forward(self, N, ratio, in_place, engine, gc, dc):
        op = core.CreateOperator(
            "Dropout", ["X"], ["Y", "mask"],
            ratio=ratio,
            engine=engine,
            is_test=False)
        X = np.ones(N).astype(np.float32)
        workspace.FeedBlob("X", X, device_option=gc)
        workspace.RunOperatorOnce(op)
        Y = workspace.FetchBlob("Y")
        np.testing.assert_allclose(np.sum(Y) / N, 1.0, rtol=0.1, atol=0)


if __name__ == "__main__":
    import unittest
    unittest.main()
