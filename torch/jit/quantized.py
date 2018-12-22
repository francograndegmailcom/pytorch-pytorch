import torch
import copy
import numbers
from typing import Tuple

from torch.nn.utils.rnn import PackedSequence


class QuantizedLinear(torch.jit.ScriptModule):
    __constants__ = ['scale', 'zero_point']

    def __init__(self, other):
        super(QuantizedLinear, self).__init__()
        self.in_features = other.in_features
        self.out_features = other.out_features
        # Quantize weight and discard the original
        self.weight, self.col_offsets, self.scale, self.zero_point = torch.fbgemm_linear_quantize_weight(
            other.weight.clone().float())
        self.weight = torch.nn.Parameter(self.weight, requires_grad=False)
        self.col_offsets = torch.nn.Parameter(self.col_offsets, requires_grad=False)
        assert other.bias is not None, 'QuantizedLinear requires a bias'
        self.bias = torch.nn.Parameter(other.bias.clone().float())

        self.register_buffer(
            'packed_tensor_ptr',
            torch.fbgemm_pack_quantized_matrix(self.weight.clone(), self.weight.size(1), self.weight.size(0)))

    @torch.jit.script_method
    def _unpack(self):
        self.packed_tensor_ptr.set_(
            torch.fbgemm_pack_quantized_matrix(
                self.weight, self.weight.size(1), self.weight.size(0)))

    @torch.jit.script_method
    def _pack(self):
        self.packed_tensor_ptr.set_(
            torch.zeros(torch.jit.annotate(List[int], []), dtype=torch.uint8).detach())

    @torch.jit.script_method
    def forward(self, input):
        out = torch.fbgemm_linear_int8_weight(
            input.float(), self.weight, self.packed_tensor_ptr, self.col_offsets,
            self.scale, self.zero_point, self.bias)
        return out.type_as(input)

    def extra_repr(self):
        repr = 'in_features={in_features}, out_features={out_features}, ' \
               'scale={scale}, zero_point={zero_point}'.format(**self.__dict__)
        return repr


# Quantized RNN cell implementations
class QuantizedRNNCellBase(torch.nn.Module):
    __constants__ = ['input_size', 'hidden_size', 'bias']

    def __init__(self, other):
        super(QuantizedRNNCellBase, self).__init__()
        self.input_size = other.input_size
        self.hidden_size = other.hidden_size
        self.bias = other.bias
        if not self.bias:
            raise ValueError("Quantized RNN cells require bias terms")

        self.weight_ih, self.col_offsets_ih, self.scale_ih, self.zero_point_ih = \
            torch.fbgemm_linear_quantize_weight(other.weight_ih.clone().float())
        self.weight_hh, self.col_offsets_hh, self.scale_hh, self.zero_point_hh = \
            torch.fbgemm_linear_quantize_weight(other.weight_hh.clone().float())

        self.packed_ih = torch.fbgemm_pack_quantized_matrix(
            self.weight_ih, self.weight_ih.size(1), self.weight_ih.size(0))
        self.packed_hh = torch.fbgemm_pack_quantized_matrix(
            self.weight_hh, self.weight_hh.size(1), self.weight_hh.size(0))

        self.bias_ih = torch.nn.Parameter(other.bias_ih.clone().float(), requires_grad=False)
        self.bias_hh = torch.nn.Parameter(other.bias_hh.clone().float(), requires_grad=False)

    def extra_repr(self):
        s = '{input_size}, {hidden_size}'
        if 'bias' in self.__dict__ and self.bias is not True:
            s += ', bias={bias}'
        if 'nonlinearity' in self.__dict__ and self.nonlinearity != "tanh":
            s += ', nonlinearity={nonlinearity}'
        return s.format(**self.__dict__)

    @torch._jit_internal.weak_script_method
    def check_forward_input(self, input):
        if input.size(1) != self.input_size:
            raise RuntimeError(
                "input has inconsistent input_size: got {}, expected {}".format(
                    input.size(1), self.input_size))

    @torch._jit_internal.weak_script_method
    def check_forward_hidden(self, input, hx, hidden_label=''):
        # type: (Tensor, Tensor, str) -> None
        if input.size(0) != hx.size(0):
            raise RuntimeError(
                "Input batch size {} doesn't match hidden{} batch size {}".format(
                    input.size(0), hidden_label, hx.size(0)))

        if hx.size(1) != self.hidden_size:
            raise RuntimeError(
                "hidden{} has inconsistent hidden_size: got {}, expected {}".format(
                    hidden_label, hx.size(1), self.hidden_size))

@torch._jit_internal.weak_module
class QuantizedRNNCell(QuantizedRNNCellBase):
    __constants__ = ['input_size', 'hidden_size', 'bias', 'nonlinearity']

    def __init__(self, other):
        super(QuantizedRNNCell, self).__init__(other)
        self.nonlinearity = other.nonlinearity

    @torch._jit_internal.weak_script_method
    def forward(self, input, hx=None):
        # type: (Tensor, Optional[Tensor]) -> Tensor
        self.check_forward_input(input)
        if hx is None:
            _hx = torch.zeros(input.size(0), self.hidden_size, dtype=input.dtype, device=input.device)
        else:
            _hx = torch.jit._unwrap_optional(hx)
        self.check_forward_hidden(input, _hx, '')
        if self.nonlinearity == "tanh":
            ret = torch._C._VariableFunctions.quantized_rnn_tanh_cell(
                input, _hx, self.weight_ih, self.weight_hh, self.bias_ih,
                self.bias_hh, self.packed_ih, self.packed_hh, self.col_offsets_ih,
                self.col_offsets_hh, self.scale_ih, self.scale_hh, self.zero_point_ih,
                self.zero_point_hh
            )
        elif self.nonlinearity == "relu":
            ret = torch._C._VariableFunctions.quantized_rnn_relu_cell(
                input, _hx, self.weight_ih, self.weight_hh, self.bias_ih,
                self.bias_hh, self.packed_ih, self.packed_hh, self.col_offsets_ih,
                self.col_offsets_hh, self.scale_ih, self.scale_hh, self.zero_point_ih,
                self.zero_point_hh
            )
        else:
            ret = input  # TODO: remove when jit supports exception flow
            raise RuntimeError(
                "Unknown nonlinearity: {}".format(self.nonlinearity))
        return ret


@torch._jit_internal.weak_module
class QuantizedLSTMCell(QuantizedRNNCellBase):
    def __init__(self, other):
        super(QuantizedLSTMCell, self).__init__(other)

    @torch._jit_internal.weak_script_method
    def forward(self, input, hx=None):
        # type: (Tensor, Optional[Tuple[Tensor, Tensor]]) -> Tuple[Tensor, Tensor]
        self.check_forward_input(input)
        if hx is None:
            zeros = torch.zeros(input.size(0), self.hidden_size, dtype=input.dtype, device=input.device)
            _hx = (zeros, zeros)
        else:
            _hx = torch.jit._unwrap_optional(hx)
        self.check_forward_hidden(input, _hx[0], '[0]')
        self.check_forward_hidden(input, _hx[1], '[1]')
        return torch._C._VariableFunctions.quantized_lstm_cell(
            input, _hx, self.weight_ih, self.weight_hh, self.bias_ih,
            self.bias_hh, self.packed_ih, self.packed_hh, self.col_offsets_ih,
            self.col_offsets_hh, self.scale_ih, self.scale_hh, self.zero_point_ih,
            self.zero_point_hh
        )

@torch._jit_internal.weak_module
class QuantizedGRUCell(QuantizedRNNCellBase):
    def __init__(self, other):
        super(QuantizedGRUCell, self).__init__(other)

    @torch._jit_internal.weak_script_method
    def forward(self, input, hx=None):
        # type: (Tensor, Optional[Tensor]) -> Tensor
        self.check_forward_input(input)
        if hx is None:
            _hx = torch.zeros(input.size(0), self.hidden_size, dtype=input.dtype, device=input.device)
        else:
            _hx = torch.jit._unwrap_optional(hx)
        self.check_forward_hidden(input, _hx, '')
        return torch._C._VariableFunctions.quantized_gru_cell(
            input, _hx, self.weight_ih, self.weight_hh, self.bias_ih,
            self.bias_hh, self.packed_ih, self.packed_hh, self.col_offsets_ih,
            self.col_offsets_hh, self.scale_ih, self.scale_hh, self.zero_point_ih,
            self.zero_point_hh
        )


def quantize_rnn_cell_modules(module):
    reassign = {}
    for name, mod in module.named_modules():
        if mod is module:
            continue
        new_mod = quantize_rnn_cell_modules(mod)
        if new_mod != mod:
            reassign[name] = new_mod
    for name, mod in reassign.items():
        setattr(module, name, mod)
    if isinstance(module, torch.nn.LSTMCell):
        return QuantizedLSTMCell(mod)
    if isinstance(module, torch.nn.GRUCell):
        return QuantizedGRUCell(mod)
    if isinstance(module, torch.nn.RNNCell):
        return QuantizedRNNCell(mod)

    return module

def quantize_linear_modules(module):
    reassign = {}
    for name, mod in module.named_modules():
        if mod is module:
            continue
        new_mod = quantize_linear_modules(mod)
        if new_mod != mod:
            reassign[name] = new_mod

    for name, mod in reassign.items():
        setattr(module, name, mod)
    if isinstance(mod, torch.nn.Linear):
        return QuantizedLinear(mod)
    return module
