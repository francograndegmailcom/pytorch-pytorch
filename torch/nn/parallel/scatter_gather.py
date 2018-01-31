import torch
from torch.autograd import Variable
from ._functions import Scatter, Gather


def scatter(inputs, target_gpus, dim=0):
    r"""
    Slices variables into approximately equal chunks and
    distributes them across given GPUs. Duplicates
    references to objects that are not variables. Does not
    support Tensors.
    """
    def scatter_var(obj):
        return Scatter.apply(target_gpus, None, dim, obj)

    def scatter_map(obj):
        if isinstance(obj, Variable):
            return scatter_var(obj)
        assert not torch.is_tensor(obj), "Tensors not supported in scatter."
        if isinstance(obj, tuple) and len(obj) > 0:
            return list(zip(*map(scatter_var, obj)))
        if isinstance(obj, list) and len(obj) > 0:
            return list(map(list, zip(*map(scatter_var, obj))))
        if isinstance(obj, dict) and len(obj) > 0:
            return list(map(type(obj), zip(*map(scatter_var, obj.items()))))
        return [obj for targets in target_gpus]

    return scatter_map(inputs)


def scatter_kwargs(inputs, kwargs, target_gpus, dim=0):
    r"""Scatter with support for kwargs dictionary"""
    inputs = scatter(inputs, target_gpus, dim) if inputs else []
    kwargs = scatter(kwargs, target_gpus, dim) if kwargs else []
    if len(inputs) < len(kwargs):
        inputs.extend([() for _ in range(len(kwargs) - len(inputs))])
    elif len(kwargs) < len(inputs):
        kwargs.extend([{} for _ in range(len(inputs) - len(kwargs))])
    inputs = tuple(inputs)
    kwargs = tuple(kwargs)
    return inputs, kwargs


def gather(outputs, target_device, dim=0):
    r"""
    Gathers variables from different GPUs on a specified device
      (-1 means the CPU).
    """
    def gather_vars(outputs):
        return Gather.apply(target_device, dim, *outputs)

    def gather_map(outputs):
        out = outputs[0]
        if isinstance(out, Variable):
            return gather_vars(outputs)
        if out is None:
            return None
        # Assuming outputs is a iterable of iterables and not
        # an iterable of iterables of iterables (or more)
        return type(out)(map(gather_vars, zip(*outputs)))

    return gather_map(outputs)
