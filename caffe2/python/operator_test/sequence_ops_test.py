from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from caffe2.python import core
from hypothesis import given
import caffe2.python.hypothesis_test_util as hu
from caffe2.proto import caffe2_pb2
import hypothesis.strategies as st
import numpy as np
from functools import partial


def _gen_test_add_padding(with_pad_data=True,
                          is_remove=False):
    def gen_with_size(args):
        lengths, inner_shape = args
        data_dim = [sum(lengths)] + inner_shape
        lengths = np.array(lengths, dtype=np.int32)
        if with_pad_data:
            return st.tuples(
                st.just(lengths),
                hu.arrays(data_dim),
                hu.arrays(inner_shape),
                hu.arrays(inner_shape))
        else:
            return st.tuples(st.just(lengths), hu.arrays(data_dim))

    min_len = 4 if is_remove else 0
    lengths = st.lists(
        st.integers(min_value=min_len, max_value=10),
        min_size=0,
        max_size=5)
    inner_shape = st.lists(
        st.integers(min_value=1, max_value=3),
        min_size=0,
        max_size=2)
    return st.tuples(lengths, inner_shape).flatmap(gen_with_size)


def _add_padding_ref(
        start_pad_width, end_pad_width,
        data, lengths, start_padding=None, end_padding=None):
    if start_padding is None:
        start_padding = np.zeros(data.shape[1:], dtype=data.dtype)
    end_padding = (
        end_padding if end_padding is not None else start_padding)
    out_size = data.shape[0] + (
        start_pad_width + end_pad_width) * len(lengths)
    out = np.ndarray((out_size,) + data.shape[1:])
    in_ptr = 0
    out_ptr = 0
    for length in lengths:
        out[out_ptr:(out_ptr + start_pad_width)] = start_padding
        out_ptr += start_pad_width
        out[out_ptr:(out_ptr + length)] = data[in_ptr:(in_ptr + length)]
        in_ptr += length
        out_ptr += length
        out[out_ptr:(out_ptr + end_pad_width)] = end_padding
        out_ptr += end_pad_width
    lengths_out = lengths + (start_pad_width + end_pad_width)
    return (out, lengths_out)


def _remove_padding_ref(start_pad_width, end_pad_width, data, lengths):
    pad_width = start_pad_width + end_pad_width
    out_size = data.shape[0] - (
        start_pad_width + end_pad_width) * len(lengths)
    out = np.ndarray((out_size,) + data.shape[1:])
    in_ptr = 0
    out_ptr = 0
    for length in lengths:
        out_length = length - pad_width
        out[out_ptr:(out_ptr + out_length)] = data[
            (in_ptr + start_pad_width):(in_ptr + length - end_pad_width)]
        in_ptr += length
        out_ptr += out_length
    lengths_out = lengths - (start_pad_width + end_pad_width)
    return (out, lengths_out)


def _gather_padding_ref(start_pad_width, end_pad_width, data, lengths):
    start_padding = np.zeros(data.shape[1:], dtype=data.dtype)
    end_padding = np.zeros(data.shape[1:], dtype=data.dtype)
    pad_width = start_pad_width + end_pad_width
    ptr = 0
    for length in lengths:
        for _ in range(start_pad_width):
            start_padding += data[ptr]
            ptr += 1
        ptr += length - pad_width
        for _ in range(end_pad_width):
            end_padding += data[ptr]
            ptr += 1
    return (start_padding, end_padding)


class TestSequenceOps(hu.HypothesisTestCase):
    @given(start_pad_width=st.integers(min_value=1, max_value=2),
           end_pad_width=st.integers(min_value=0, max_value=2),
           args=_gen_test_add_padding(with_pad_data=True))
    def test_add_padding(self, start_pad_width, end_pad_width, args):
        lengths, data, start_padding, end_padding = args
        start_padding = np.array(start_padding, dtype=np.float32)
        end_padding = np.array(end_padding, dtype=np.float32)
        op = core.CreateOperator(
            'AddPadding',
            ['data', 'lengths', 'start_padding', 'end_padding'],
            ['output', 'lengths_out'],
            padding_width=start_pad_width,
            end_padding_width=end_pad_width)
        self.assertReferenceChecks(
            hu.cpu_do,
            op,
            [data, lengths, start_padding, end_padding],
            partial(_add_padding_ref, start_pad_width, end_pad_width))

    @given(start_pad_width=st.integers(min_value=1, max_value=2),
           end_pad_width=st.integers(min_value=0, max_value=2),
           args=_gen_test_add_padding(with_pad_data=False))
    def test_add_zero_padding(self, start_pad_width, end_pad_width, args):
        lengths, data = args
        op = core.CreateOperator(
            'AddPadding',
            ['data', 'lengths'],
            ['output', 'lengths_out'],
            padding_width=start_pad_width,
            end_padding_width=end_pad_width)
        self.assertReferenceChecks(
            hu.cpu_do,
            op,
            [data, lengths],
            partial(_add_padding_ref, start_pad_width, end_pad_width))

    @given(start_pad_width=st.integers(min_value=1, max_value=2),
           end_pad_width=st.integers(min_value=0, max_value=2),
           data=hu.tensor(min_dim=1, max_dim=3))
    def test_add_padding_no_length(self, start_pad_width, end_pad_width, data):
        op = core.CreateOperator(
            'AddPadding',
            ['data'],
            ['output', 'output_lens'],
            padding_width=start_pad_width,
            end_padding_width=end_pad_width)
        self.assertReferenceChecks(
            hu.cpu_do,
            op,
            [data],
            partial(
                _add_padding_ref, start_pad_width, end_pad_width,
                lengths=np.array([data.shape[0]])))

    @given(start_pad_width=st.integers(min_value=1, max_value=2),
           end_pad_width=st.integers(min_value=0, max_value=2),
           args=_gen_test_add_padding(with_pad_data=False, is_remove=True))
    def test_remove_padding(self, start_pad_width, end_pad_width, args):
        lengths, data = args
        op = core.CreateOperator(
            'RemovePadding',
            ['data', 'lengths'],
            ['output', 'lengths_out'],
            padding_width=start_pad_width,
            end_padding_width=end_pad_width)
        self.assertReferenceChecks(
            hu.cpu_do,
            op,
            [data, lengths],
            partial(_remove_padding_ref, start_pad_width, end_pad_width))

    @given(start_pad_width=st.integers(min_value=1, max_value=2),
           end_pad_width=st.integers(min_value=0, max_value=2),
           args=_gen_test_add_padding(with_pad_data=True))
    def test_gather_padding(self, start_pad_width, end_pad_width, args):
        lengths, data, start_padding, end_padding = args
        padded_data, padded_lengths = _add_padding_ref(
            start_pad_width, end_pad_width, data,
            lengths, start_padding, end_padding)
        op = core.CreateOperator(
            'GatherPadding',
            ['data', 'lengths'],
            ['start_padding', 'end_padding'],
            padding_width=start_pad_width,
            end_padding_width=end_pad_width)
        self.assertReferenceChecks(
            hu.cpu_do,
            op,
            [padded_data, padded_lengths],
            partial(_gather_padding_ref, start_pad_width, end_pad_width))

    @given(data=hu.tensor(min_dim=3, max_dim=3, dtype=np.float32,
                          elements=st.floats(min_value=-np.inf,
                                             max_value=np.inf),
                          min_value=1, max_value=10),
                          **hu.gcs)
    def test_reverse_packed_segs(self, data, gc, dc):
        max_length = data.shape[0]
        batch_size = data.shape[1]
        lengths = np.random.randint(max_length + 1, size=batch_size)

        op = core.CreateOperator(
            "ReversePackedSegs",
            ["data", "lengths"],
            ["reversed_data"])

        def op_ref(data, lengths):
            rev_data = np.array(data, copy=True)
            for i in range(batch_size):
                seg_length = lengths[i]
                for j in range(seg_length):
                    rev_data[j][i] = data[seg_length - 1 - j][i]
            return (rev_data,)

        def op_grad_ref(grad_out, outputs, inputs):
            return op_ref(grad_out, inputs[1]) + (None,)

        self.assertReferenceChecks(
            device_option=gc,
            op=op,
            inputs=[data, lengths],
            reference=op_ref,
            input_device_options={"lengths": core.DeviceOption(caffe2_pb2.CPU)},
            output_to_grad='reversed_data',
            grad_reference=op_grad_ref)

    @given(data=hu.tensor(min_dim=1, max_dim=3, dtype=np.float32,
                          elements=st.floats(min_value=-np.inf,
                                             max_value=np.inf),
                          min_value=10, max_value=10),
           indices=st.lists(st.integers(min_value=0, max_value=9),
                            min_size=0,
                            max_size=10),
           **hu.gcs_cpu_only)
    def test_remove_data_blocks(self, data, indices, gc, dc):
        indices = np.array(indices)

        op = core.CreateOperator(
            "RemoveDataBlocks",
            ["data", "indices"],
            ["shrunk_data"])

        def op_ref(data, indices):
            unique_indices = np.unique(indices)
            sorted_indices = np.sort(unique_indices)
            shrunk_data = np.delete(data, sorted_indices, axis=0)
            return (shrunk_data,)

        self.assertReferenceChecks(
            device_option=gc,
            op=op,
            inputs=[data, indices],
            reference=op_ref)

    @given(elements=st.lists(st.integers(min_value=0, max_value=9),
                             min_size=0,
                             max_size=10),
           **hu.gcs_cpu_only)
    def test_find_duplicate_elements(self, elements, gc, dc):
        mapping = {
            0: "a",
            1: "b",
            2: "c",
            3: "d",
            4: "e",
            5: "f",
            6: "g",
            7: "h",
            8: "i",
            9: "j"}
        data = np.array([mapping[e] for e in elements], dtype='|S')

        op = core.CreateOperator(
            "FindDuplicateElements",
            ["data"],
            ["indices"])

        def op_ref(data):
            unique_data = []
            indices = []
            for i, e in enumerate(data):
                if e in unique_data:
                    indices.append(i)
                else:
                    unique_data.append(e)
            return (np.array(indices, dtype=np.int64),)

        self.assertReferenceChecks(
            device_option=gc,
            op=op,
            inputs=[data],
            reference=op_ref)

if __name__ == "__main__":
    import unittest
    unittest.main()
