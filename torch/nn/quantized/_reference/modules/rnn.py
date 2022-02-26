import torch
import torch.nn as nn
from torch import Tensor
from .utils import _quantize_and_dequantize_weight
from typing import Optional, Dict, Any, Tuple
from torch import _VF
from torch.nn.utils.rnn import PackedSequence

class RNNCellBase(nn.RNNCellBase):
    def __init__(self, input_size: int, hidden_size: int, bias: bool, num_chunks: int,
                 device=None, dtype=None, weight_qparams_dict=None) -> None:
        super().__init__(input_size, hidden_size, bias, num_chunks, device=device, dtype=dtype)
        if weight_qparams_dict is None:
            weight_qparams = {
                "qscheme": torch.per_tensor_affine,
                "dtype": torch.quint8,
                "scale": 1.0,
                "zero_point": 0
            }
            weight_qparams_dict = {
                "weight_ih": weight_qparams,
                "weight_hh": weight_qparams
            }
        assert len(weight_qparams_dict) == 2, "Expected length for weight_qparams_dict to be 2 for QuantizedRNNCellBase(Reference)"
        self._init_weight_qparams_dict(weight_qparams_dict, device)

    def _init_weight_qparams_dict(self, weight_qparams_dict, device):
        assert weight_qparams_dict is not None
        for key, weight_qparams in weight_qparams_dict.items():
            # TODO: refactor the duplicated code to utils.py
            weight_qscheme = weight_qparams["qscheme"]
            weight_dtype = weight_qparams["dtype"]
            setattr(self, key + "_qscheme", weight_qscheme)
            setattr(self, key + "_dtype", weight_dtype)
            assert weight_qscheme in [None, torch.per_tensor_affine, torch.per_channel_affine], \
                Exception(f"qscheme: {weight_qscheme} is not support in {self._get_name()}")
            if weight_qscheme is not None:
                self.register_buffer(
                    key + "_scale",
                    torch.tensor(weight_qparams["scale"], dtype=torch.float, device=device))
                self.register_buffer(
                    key + "_zero_point",
                    torch.tensor(weight_qparams["zero_point"], dtype=torch.int, device=device))
                if weight_qscheme == torch.per_channel_affine:
                    self.register_buffer(
                        key + "_axis",
                        torch.tensor(weight_qparams["axis"], dtype=torch.int, device=device))
                else:
                    # added for TorchScriptability, not used
                    self.register_buffer(
                        key + "_axis", torch.tensor(0, dtype=torch.int, device=device))

    def _get_name(self):
        return "QuantizedRNNCellBase(Reference)"

    def get_weight_ih(self):
        wn = "weight_ih"
        weight = self.weight_ih
        weight_qscheme = getattr(self, wn + "_qscheme")
        weight_dtype = getattr(self, wn + "_dtype")
        weight_scale = getattr(self, wn + "_scale")
        weight_zero_point = getattr(self, wn + "_zero_point")
        weight_axis = getattr(self, wn + "_axis")
        weight = _quantize_and_dequantize_weight(weight, weight_qscheme, weight_dtype, weight_scale, weight_zero_point, weight_axis)
        return weight

    def get_weight_hh(self):
        wn = "weight_hh"
        weight = self.weight_hh
        weight_qscheme = getattr(self, wn + "_qscheme")
        weight_dtype = getattr(self, wn + "_dtype")
        weight_scale = getattr(self, wn + "_scale")
        weight_zero_point = getattr(self, wn + "_zero_point")
        weight_axis = getattr(self, wn + "_axis")
        weight = _quantize_and_dequantize_weight(weight, weight_qscheme, weight_dtype, weight_scale, weight_zero_point, weight_axis)
        return weight

class RNNCell(RNNCellBase):
    """
    We'll store weight_qparams for all the weights (weight_ih and weight_hh),
    we need to pass in a `weight_qparams_dict` that maps from weight name,
    e.g. weight_ih, to the weight_qparams for that weight
    """
    def __init__(self, input_size: int, hidden_size: int, bias: bool = True, nonlinearity: str = "tanh",
                 device=None, dtype=None, weight_qparams_dict: Optional[Dict[str, Dict[str, Any]]] = None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype, 'weight_qparams_dict': weight_qparams_dict}
        super().__init__(input_size, hidden_size, bias, num_chunks=1, **factory_kwargs)
        self.nonlinearity = nonlinearity

    def _get_name(self):
        return "QuantizedRNNCell(Reference)"

    # TODO: refactor nn.RNNCell to have a _forward that takes weight_ih and weight_hh as input
    # and remove duplicated code, same for the other two Cell modules
    def forward(self, input: Tensor, hx: Optional[Tensor] = None) -> Tensor:
        assert input.dim() in (1, 2), \
            f"RNNCell: Expected input to be 1-D or 2-D but received {input.dim()}-D tensor"
        is_batched = input.dim() == 2
        if not is_batched:
            input = input.unsqueeze(0)

        if hx is None:
            hx = torch.zeros(input.size(0), self.hidden_size, dtype=input.dtype, device=input.device)
        else:
            hx = hx.unsqueeze(0) if not is_batched else hx

        if self.nonlinearity == "tanh":
            ret = _VF.rnn_tanh_cell(
                input, hx,
                self.get_weight_ih(), self.get_weight_hh(),
                self.bias_ih, self.bias_hh,
            )
        elif self.nonlinearity == "relu":
            ret = _VF.rnn_relu_cell(
                input, hx,
                self.get_weight_ih(), self.get_weight_hh(),
                self.bias_ih, self.bias_hh,
            )
        else:
            ret = input  # TODO: remove when jit supports exception flow
            raise RuntimeError(
                "Unknown nonlinearity: {}".format(self.nonlinearity))

        if not is_batched:
            ret = ret.squeeze(0)

        return ret


class LSTMCell(RNNCellBase):
    """
    We'll store weight_qparams for all the weights (weight_ih and weight_hh),
    we need to pass in a `weight_qparams_dict` that maps from weight name,
    e.g. weight_ih, to the weight_qparams for that weight
    """
    def __init__(self, input_size: int, hidden_size: int, bias: bool = True, nonlinearity: str = "tanh",
                 device=None, dtype=None, weight_qparams_dict: Optional[Dict[str, Dict[str, Any]]] = None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype, 'weight_qparams_dict': weight_qparams_dict}
        super().__init__(input_size, hidden_size, bias, num_chunks=4, **factory_kwargs)
        self.nonlinearity = nonlinearity

    def _get_name(self):
        return "QuantizedLSTMCell(Reference)"

    def forward(self, input: Tensor, hx: Optional[Tuple[Tensor, Tensor]] = None) -> Tuple[Tensor, Tensor]:
        assert input.dim() in (1, 2), \
            f"LSTMCell: Expected input to be 1-D or 2-D but received {input.dim()}-D tensor"
        is_batched = input.dim() == 2
        if not is_batched:
            input = input.unsqueeze(0)

        if hx is None:
            zeros = torch.zeros(input.size(0), self.hidden_size, dtype=input.dtype, device=input.device)
            hx = (zeros, zeros)
        else:
            hx = (hx[0].unsqueeze(0), hx[1].unsqueeze(0)) if not is_batched else hx

        ret = _VF.lstm_cell(
            input, hx,
            self.get_weight_ih(), self.get_weight_hh(),
            self.bias_ih, self.bias_hh,
        )

        if not is_batched:
            ret = (ret[0].squeeze(0), ret[1].squeeze(0))
        return ret



class GRUCell(RNNCellBase):
    """
    We'll store weight_qparams for all the weights (weight_ih and weight_hh),
    we need to pass in a `weight_qparams_dict` that maps from weight name,
    e.g. weight_ih, to the weight_qparams for that weight
    """
    def __init__(self, input_size: int, hidden_size: int, bias: bool = True, nonlinearity: str = "tanh",
                 device=None, dtype=None, weight_qparams_dict: Optional[Dict[str, Dict[str, Any]]] = None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype, 'weight_qparams_dict': weight_qparams_dict}
        super().__init__(input_size, hidden_size, bias, num_chunks=3, **factory_kwargs)
        self.nonlinearity = nonlinearity

    def _get_name(self):
        return "QuantizedGRUCell(Reference)"

    def forward(self, input: Tensor, hx: Optional[Tensor] = None) -> Tensor:
        assert input.dim() in (1, 2), \
            f"GRUCell: Expected input to be 1-D or 2-D but received {input.dim()}-D tensor"
        is_batched = input.dim() == 2
        if not is_batched:
            input = input.unsqueeze(0)

        if hx is None:
            hx = torch.zeros(input.size(0), self.hidden_size, dtype=input.dtype, device=input.device)
        else:
            hx = hx.unsqueeze(0) if not is_batched else hx

        ret = _VF.gru_cell(
            input, hx,
            self.get_weight_ih(), self.get_weight_hh(),
            self.bias_ih, self.bias_hh,
        )

        if not is_batched:
            ret = ret.squeeze(0)

        return ret


class RNNBase(nn.RNNBase):
    def __init__(self, mode: str, input_size: int, hidden_size: int,
                 num_layers: int = 1, bias: bool = True, batch_first: bool = False,
                 dropout: float = 0., bidirectional: bool = False, proj_size: int = 0,
                 device=None, dtype=None, weight_qparams_dict=None) -> None:
        super().__init__(
            mode, input_size, hidden_size, num_layers, bias, batch_first, dropout,
            bidirectional, proj_size, device, dtype
        )
        if weight_qparams_dict is None:
            weight_qparams = {
                'qscheme': torch.per_tensor_affine,
                'dtype': torch.quint8,
                'scale': 1.0,
                'zero_point': 0
            }
            for wn in self._flat_weights_names:
                weight_qparams_dict[wn] = weight_qparams

        self._init_weight_qparams_dict(weight_qparams_dict, device)

    def _init_weight_qparams_dict(self, weight_qparams_dict, device):
        for key, weight_qparams in weight_qparams_dict.items():
            weight_qscheme = weight_qparams["qscheme"]
            weight_dtype = weight_qparams["dtype"]
            setattr(self, key + "_qscheme", weight_qscheme)
            setattr(self, key + "_dtype", weight_dtype)
            assert weight_qscheme in [None, torch.per_tensor_affine, torch.per_channel_affine], \
                Exception(f"qscheme: {weight_qscheme} is not support in {self._get_name()}")
            if weight_qscheme is not None:
                self.register_buffer(
                    key + "_scale",
                    torch.tensor(weight_qparams["scale"], dtype=torch.float, device=device))
                self.register_buffer(
                    key + "_zero_point",
                    torch.tensor(weight_qparams["zero_point"], dtype=torch.int, device=device))
                if weight_qscheme == torch.per_channel_affine:
                    self.register_buffer(
                        key + "_axis",
                        torch.tensor(weight_qparams["axis"], dtype=torch.int, device=device))
                else:
                    # added for TorchScriptability, not used
                    self.register_buffer(
                        key + "_axis", torch.tensor(0, dtype=torch.int, device=device))

class LSTM(RNNBase):
    """ Reference Quantized LSTM Module
    We'll store weight_qparams for all the weights in _flat_weights, we need to pass in
    a `weight_qparams_dict` that maps from weight name, e.g. weight_ih_l0,
    to the weight_qparams for that weight
    """
    def __init__(self, *args, **kwargs):
        assert "weight_qparams_dict" in kwargs
        super(LSTM, self).__init__('LSTM', *args, **kwargs)

    def get_expected_cell_size(self, input: Tensor, batch_sizes: Optional[Tensor]) -> Tuple[int, int, int]:
        if batch_sizes is not None:
            mini_batch = int(batch_sizes[0])
        else:
            mini_batch = input.size(0) if self.batch_first else input.size(1)
        num_directions = 2 if self.bidirectional else 1
        expected_hidden_size = (self.num_layers * num_directions,
                                mini_batch, self.hidden_size)
        return expected_hidden_size

    # In the future, we should prevent mypy from applying contravariance rules here.
    # See torch/nn/modules/module.py::_forward_unimplemented
    def check_forward_args(self,  # type: ignore[override]
                           input: Tensor,
                           hidden: Tuple[Tensor, Tensor],
                           batch_sizes: Optional[Tensor],
                           ):
        self.check_input(input, batch_sizes)
        self.check_hidden_size(hidden[0], self.get_expected_hidden_size(input, batch_sizes),
                               'Expected hidden[0] size {}, got {}')
        self.check_hidden_size(hidden[1], self.get_expected_cell_size(input, batch_sizes),
                               'Expected hidden[1] size {}, got {}')

    def get_flat_weights(self):
        flat_weights = []
        for wn in self._flat_weights_names:
            if hasattr(self, wn):
                weight = getattr(self, wn)
                weight_qscheme = getattr(self, wn + "_qscheme")
                weight_dtype = getattr(self, wn + "_dtype")
                weight_scale = getattr(self, wn + "_scale")
                weight_zero_point = getattr(self, wn + "_zero_point")
                weight_axis = getattr(self, wn + "_axis")
                weight = _quantize_and_dequantize_weight(weight, weight_qscheme, weight_dtype, weight_scale, weight_zero_point, weight_axis)
            else:
                weight = None
            flat_weights.append(weight)
        return flat_weights

    def forward(self, input, hx=None):  # noqa: F811
        orig_input = input
        # xxx: isinstance check needs to be in conditional for TorchScript to compile
        batch_sizes = None
        if isinstance(orig_input, PackedSequence):
            input, batch_sizes, sorted_indices, unsorted_indices = input
            max_batch_size = batch_sizes[0]
            max_batch_size = int(max_batch_size)
        else:
            batch_sizes = None
            is_batched = input.dim() == 3
            batch_dim = 0 if self.batch_first else 1
            if not is_batched:
                input = input.unsqueeze(batch_dim)
            max_batch_size = input.size(0) if self.batch_first else input.size(1)
            sorted_indices = None
            unsorted_indices = None

        if hx is None:
            num_directions = 2 if self.bidirectional else 1
            real_hidden_size = self.proj_size if self.proj_size > 0 else self.hidden_size
            h_zeros = torch.zeros(self.num_layers * num_directions,
                                  max_batch_size, real_hidden_size,
                                  dtype=input.dtype, device=input.device)
            c_zeros = torch.zeros(self.num_layers * num_directions,
                                  max_batch_size, self.hidden_size,
                                  dtype=input.dtype, device=input.device)
            hx = (h_zeros, c_zeros)
        else:
            if batch_sizes is None:  # If not PackedSequence input.
                if is_batched:
                    if (hx[0].dim() != 3 or hx[1].dim() != 3):
                        msg = ("For batched 3-D input, hx and cx should "
                               f"also be 3-D but got ({hx[0].dim()}-D, {hx[1].dim()}-D) tensors")
                        raise RuntimeError(msg)
                else:
                    if hx[0].dim() != 2 or hx[1].dim() != 2:
                        msg = ("For unbatched 2-D input, hx and cx should "
                               f"also be 2-D but got ({hx[0].dim()}-D, {hx[1].dim()}-D) tensors")
                        raise RuntimeError(msg)
                    hx = (hx[0].unsqueeze(1), hx[1].unsqueeze(1))

            # Each batch of the hidden state should match the input sequence that
            # the user believes he/she is passing in.
            hx = self.permute_hidden(hx, sorted_indices)

        self.check_forward_args(input, hx, batch_sizes)
        if batch_sizes is None:
            result = _VF.lstm(input, hx, self.get_flat_weights(), self.bias, self.num_layers,
                              self.dropout, self.training, self.bidirectional, self.batch_first)
        else:
            result = _VF.lstm(input, batch_sizes, hx, self.get_flat_weights(), self.bias,
                              self.num_layers, self.dropout, self.training, self.bidirectional)
        output = result[0]
        hidden = result[1:]
        # xxx: isinstance check needs to be in conditional for TorchScript to compile
        if isinstance(orig_input, PackedSequence):
            output_packed = PackedSequence(output, batch_sizes, sorted_indices, unsorted_indices)
            return output_packed, self.permute_hidden(hidden, unsorted_indices)
        else:
            if not is_batched:
                output = output.squeeze(batch_dim)
                hidden = (hidden[0].squeeze(1), hidden[1].squeeze(1))
            return output, self.permute_hidden(hidden, unsorted_indices)

    def _get_name(self):
        return "QuantizedLSTM(Reference)"
