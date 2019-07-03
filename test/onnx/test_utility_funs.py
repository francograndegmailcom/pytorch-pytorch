from __future__ import absolute_import, division, print_function, unicode_literals
from test_pytorch_common import TestCase, run_tests

import torch
import torch.onnx
from torch.onnx import utils
from torch.onnx.symbolic_helper import _set_opset_version

import onnx

import io
import copy


class TestUtilityFuns(TestCase):
    from torch.onnx.symbolic_helper import _export_onnx_opset_version
    opset_version = _export_onnx_opset_version

    def test_is_in_onnx_export(self):
        test_self = self

        class MyModule(torch.nn.Module):
            def forward(self, x):
                test_self.assertTrue(torch.onnx.is_in_onnx_export())
                raise ValueError
                return x + 1

        x = torch.randn(3, 4)
        f = io.BytesIO()
        try:
            torch.onnx.export(MyModule(), x, f, opset_version=self.opset_version)
        except ValueError:
            self.assertFalse(torch.onnx.is_in_onnx_export())

    def test_constant_fold_transpose(self):
        class TransposeModule(torch.nn.Module):
            def forward(self, x):
                a = torch.tensor([[1., 2., 3.], [4., 5., 6.]])
                b = torch.transpose(a, 1, 0)
                return b + x

        _set_opset_version(self.opset_version)
        x = torch.ones(3, 2)
        graph, _, __ = utils._model_to_graph(TransposeModule(), (x, ),
                                             do_constant_folding=True,
                                             _disable_torch_constant_prop=True)
        for node in graph.nodes():
            assert node.kind() != "onnx::Transpose"
            assert node.kind() != "onnx::Cast"
            assert node.kind() != "onnx::Constant"
        assert len(list(graph.nodes())) == 1

    def test_constant_fold_slice(self):
        class NarrowModule(torch.nn.Module):
            def forward(self, x):
                a = torch.tensor([[1., 2., 3.], [4., 5., 6.]])
                b = torch.narrow(a, 0, 0, 1)
                return b + x

        _set_opset_version(self.opset_version)
        x = torch.ones(1, 3)
        graph, _, __ = utils._model_to_graph(NarrowModule(), (x, ),
                                             do_constant_folding=True,
                                             _disable_torch_constant_prop=True)
        for node in graph.nodes():
            assert node.kind() != "onnx::Slice"
            assert node.kind() != "onnx::Cast"
            assert node.kind() != "onnx::Constant"
        assert len(list(graph.nodes())) == 1

    def test_constant_fold_slice_index_exceeds_dim(self):
        class SliceIndexExceedsDimModule(torch.nn.Module):
            def forward(self, x):
                a = torch.tensor([[1., 2., 3.], [4., 5., 6.]])
                b = a[1:10]         # index exceeds dimension
                return b + x

        _set_opset_version(self.opset_version)
        x = torch.ones(1, 3)
        graph, _, __ = utils._model_to_graph(SliceIndexExceedsDimModule(), (x, ),
                                             do_constant_folding=True,
                                             _disable_torch_constant_prop=True)

        for node in graph.nodes():
            assert node.kind() != "onnx::Slice"
            assert node.kind() != "onnx::Cast"
            assert node.kind() != "onnx::Constant"
        assert len(list(graph.nodes())) == 1

    def test_constant_fold_slice_negative_index(self):
        class SliceNegativeIndexModule(torch.nn.Module):
            def forward(self, x):
                a = torch.tensor([[1., 2., 3.], [4., 5., 6.]])
                b = a[0:-1]        # index relative to the end
                return b + x

        _set_opset_version(self.opset_version)
        x = torch.ones(1, 3)
        graph, _, __ = utils._model_to_graph(SliceNegativeIndexModule(), (x, ),
                                             do_constant_folding=True,
                                             _disable_torch_constant_prop=True)
        for node in graph.nodes():
            assert node.kind() != "onnx::Slice"
            assert node.kind() != "onnx::Cast"
            assert node.kind() != "onnx::Constant"
        assert len(list(graph.nodes())) == 1

    def test_constant_fold_unsqueeze(self):
        class UnsqueezeModule(torch.nn.Module):
            def forward(self, x):
                a = torch.tensor([[1., 2., 3.], [4., 5., 6.]])
                b = torch.unsqueeze(a, 0)
                return b + x

        _set_opset_version(self.opset_version)
        x = torch.ones(1, 2, 3)
        graph, _, __ = utils._model_to_graph(UnsqueezeModule(), (x, ),
                                             do_constant_folding=True,
                                             _disable_torch_constant_prop=True)
        for node in graph.nodes():
            assert node.kind() != "onnx::Unsqueeeze"
            assert node.kind() != "onnx::Cast"
            assert node.kind() != "onnx::Constant"
        assert len(list(graph.nodes())) == 1

    def test_constant_fold_concat(self):
        class ConcatModule(torch.nn.Module):
            def forward(self, x):
                a = torch.tensor([[1., 2., 3.]])
                b = torch.tensor([[4., 5., 6.]])
                c = torch.cat((a, b), 0)
                return b + c

        _set_opset_version(self.opset_version)
        x = torch.ones(2, 3)
        graph, _, __ = utils._model_to_graph(ConcatModule(), (x, ),
                                             do_constant_folding=True,
                                             _disable_torch_constant_prop=True)
        for node in graph.nodes():
            assert node.kind() != "onnx::Concat"
            assert node.kind() != "onnx::Cast"
            assert node.kind() != "onnx::Constant"
        assert len(list(graph.nodes())) == 1

    def test_constant_fold_lstm(self):
        class GruNet(torch.nn.Module):
            def __init__(self):
                super(GruNet, self).__init__()
                self.mygru = torch.nn.GRU(7, 3, 1, bidirectional=False)

            def forward(self, input, initial_state):
                return self.mygru(input, initial_state)

        _set_opset_version(self.opset_version)
        input = torch.randn(5, 3, 7)
        h0 = torch.randn(1, 3, 3)
        graph, _, __ = utils._model_to_graph(GruNet(), (input, h0),
                                             do_constant_folding=True)
        for node in graph.nodes():
            assert node.kind() != "onnx::Slice"
            assert node.kind() != "onnx::Concat"
            assert node.kind() != "onnx::Unsqueeze"
        assert len(list(graph.nodes())) == 3

    def test_constant_fold_transpose_matmul(self):
        class MatMulNet(torch.nn.Module):
            def __init__(self):
                super(MatMulNet, self).__init__()
                self.B = torch.nn.Parameter(torch.ones(5, 3))

            def forward(self, A):
                return torch.matmul(A, torch.transpose(self.B, -1, -2))

        _set_opset_version(self.opset_version)
        A = torch.randn(2, 3)
        graph, _, __ = utils._model_to_graph(MatMulNet(), (A),
                                             do_constant_folding=True)
        for node in graph.nodes():
            assert node.kind() != "onnx::Transpose"
        assert len(list(graph.nodes())) == 1

    def test_strip_doc_string(self):
        class MyModule(torch.nn.Module):
            def forward(self, input):
                return torch.exp(input)
        x = torch.randn(3, 4)

        def is_model_stripped(f, strip_doc_string=None):
            if strip_doc_string is None:
                torch.onnx.export(MyModule(), x, f, opset_version=self.opset_version)
            else:
                torch.onnx.export(MyModule(), x, f, strip_doc_string=strip_doc_string,
                                  opset_version=self.opset_version)
            model = onnx.load(io.BytesIO(f.getvalue()))
            model_strip = copy.copy(model)
            onnx.helper.strip_doc_string(model_strip)
            return model == model_strip

        # test strip_doc_string=True (default)
        self.assertTrue(is_model_stripped(io.BytesIO()))
        # test strip_doc_string=False
        self.assertFalse(is_model_stripped(io.BytesIO(), False))

# opset 10 tests
TestUtilityFuns_opset10 = type(str("TestUtilityFuns_opset10"),
                               (TestCase,),
                               dict(TestUtilityFuns.__dict__, opset_version=10))

if __name__ == '__main__':
    run_tests()
