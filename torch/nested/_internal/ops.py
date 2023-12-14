import functools
import math

import torch
from torch.nested._internal.sdpa import jagged_scaled_dot_product_attention

from .nested_tensor import NestedTensor
from typing import *  # noqa: F403
from torch.fx.operator_schemas import normalize_function

__all__: List[Any] = []

JAGGED_OPS_TABLE: Dict[Any, Any] = {}


# Simplifying assumption: we assume that the batch dim is always the left-most
# dim, and the ragged dim is always the second dim.
def _outer_to_inner_dim(ndim, dim):
    assert dim >= 0 and dim < ndim
    return 0 if dim < 2 else dim - 1


def _wrap_jagged_dim(ndim, dim, op_name):
    from torch._prims_common import canonicalize_dims

    wrapped = canonicalize_dims(ndim, dim)
    if wrapped < 2:
        raise RuntimeError(
            f"{op_name}(): not supported for NestedTensor on dim=0 or dim=1"
        )
    return _outer_to_inner_dim(ndim, wrapped)


def _wrap_jagged_dims(ndim, dims, op_name):
    # ex: (2, 3, 4) -> (1, 2, 3)
    # ex: (0, 1, 4) -> (0, 3)
    from torch._prims_common import canonicalize_dims

    wrapped_dims = [canonicalize_dims(ndim, d) for d in dims]
    # This logic needs to be done after we canonicalize dims but before we
    # map to inner dims so we can print a nicer error message.
    zero_in_dims = 0 in wrapped_dims
    one_in_dims = 1 in wrapped_dims
    if zero_in_dims ^ one_in_dims:
        apply, not_apply = ("batch", "ragged") if zero_in_dims else ("ragged", "batch")
        raise RuntimeError(
            f"{op_name}(): applying over the {apply} dimension, but not the {not_apply}"
            " dimension is not supported for NestedTensor"
        )
    return (
        tuple(_outer_to_inner_dim(ndim, d) for d in dims if d != 0),
        zero_in_dims,
    )


def check_schema(schema_str: str, func, *args, **kwargs) -> None:
    named_arg_types = schema_str.split(", ")
    num_optional_args = sum([x.endswith("?") for x in named_arg_types])
    min_args = len(named_arg_types) - num_optional_args

    # special case: ellipses allows for any number of unchecked args at the end
    if named_arg_types[-1] == "...":
        named_arg_types = named_arg_types[:-1]
    else:
        if not (len(args) >= min_args and len(args) <= len(named_arg_types)):
            raise ValueError(
                f"NestedTensor {func.__name__}({schema_str}): expected at least {min_args} "
                f"arguments and at most {len(named_arg_types)} arguments, but got: "
                f"{len(args)} arguments"
            )

    arg_type_check_fns = {
        "t": lambda x: isinstance(x, torch.Tensor) and not isinstance(x, NestedTensor),
        "jt": lambda x: isinstance(x, NestedTensor)
        and x._lengths is None
        and x._ragged_idx == 1,  # ops with "jt" require contiguous JT only
        "jt_all": lambda x: isinstance(
            x, NestedTensor
        ),  # ops with "jt_all" can accept all kinds of JT
        "any": lambda x: True,
    }
    for i, named_arg_type in enumerate(named_arg_types):
        name, arg_type = named_arg_type.split(": ")
        is_optional = arg_type.endswith("?")
        normalized_arg_type = arg_type[:-1] if is_optional else arg_type
        if normalized_arg_type not in arg_type_check_fns.keys():
            raise AssertionError(f"Unknown arg type: {normalized_arg_type}")

        if i >= len(args):
            if not is_optional:
                raise ValueError(
                    f"NestedTensor {func.__name__}({schema_str}) "
                    f"missing required argument: {name}"
                )
            continue

        if not arg_type_check_fns[normalized_arg_type](args[i]):
            type_to_desc = {
                "t": "tensor",
                "jt": "contiguous jagged layout NestedTensor",
                "jt_all": "jagged layout NestedTensor",
                "any": "<any type>",
            }

            raise ValueError(
                f"NestedTensor {func.__name__}({schema_str}): expected {name} to be a "
                f"{type_to_desc[arg_type]}"
            )


def check_ragged_dim_same(
    func, a: NestedTensor, a_name: str, b: NestedTensor, b_name: str
) -> None:
    # Calling into .shape here
    if a._size[a._ragged_idx] != b._size[b._ragged_idx]:
        raise RuntimeError(
            f"NestedTensor {func.__name__}: expected {a_name} and {b_name} to have the "
            "same exact offsets tensor."
        )


def validate_and_get_meta(func_name, size):
    # Assume that there's only a single ragged dimension
    _unused_B, singleton, *Ds = size
    if not (isinstance(singleton, torch.SymInt) and singleton.node.is_singleton()):
        raise ValueError(
            f"{func_name}() only supports shapes of form (B, *, D1, D2...) "
            "where only the second-left-most dimension is ragged. "
        )
    offsets = singleton.node.singleton_data()
    sum_offsets = singleton.node.singleton_sum_offsets()
    return offsets, sum_offsets, *Ds


# returns True if the raggedness-relevant portions of the NT shape
# match those of the specified size
def raggedness_matches(nt, size):
    end = nt._ragged_idx + 1
    return list(nt._size[:end]) == list(size[:end])


def squeeze_leading_ones(t):
    # Note: [ Squeezing leading ones ]
    #
    # Squeeze leading ones from t.
    #
    # We want:
    #   (B, j0, ?, ?) + (1, 1, ?, ?) -> (B, j0, ?, ?)
    #   (B, j0, ?, ?) + (1, 1, 1, ?, ?) -> (1, B, j0, ?, ?)  (not yet supported)
    #
    # 1) Squeeze extra ones and grab values from NT
    #   (1, 1, ?, ?) -> (?, ?)   and   (sum(*), ?, ?) -> (B, j0, ?, ?)
    # 2) Do dense broadcasting:
    #   (sum(*), ?, ?) + (?, ?) -> (sum(*), ?, ?)
    # 3) Construct nested tensor
    #   (sum(*), ?, ?) -> (B, j0, ?, ?)
    #
    # If unsqueezing on the 0th dim becomes supported, we would unsqueeze
    # at step (4) and we would need to update this function to record how
    # many ones we unsqueezed.
    while t.shape[0] == 1:
        t = t.squeeze(0)
    return t


def register_func(tables, aten_ops, schema_str):
    if not isinstance(aten_ops, list):
        aten_ops = [aten_ops]
    if not isinstance(tables, list):
        tables = [tables]

    def wrapper(func):
        for aten_op in aten_ops:

            def get_inner(aten_op):
                def inner(*args, **kwargs):
                    check_schema(schema_str, func, *args, **kwargs)
                    return func(aten_op, *args, **kwargs)

                return inner

            for table in tables:
                table[aten_op] = get_inner(aten_op)
        return func

    return wrapper


register_jagged_func = functools.partial(register_func, JAGGED_OPS_TABLE)


def lookup_jagged(func, *args, **kwargs) -> Optional[Callable]:
    dispatch_func = JAGGED_OPS_TABLE.get(func, None)
    if dispatch_func is not None:
        return dispatch_func

    # Handle pointwise fallbacks
    if torch.Tag.pointwise in func.tags:
        # Assume there aren't additional tensors that aren't the "unary/binary" args
        num_tensor_args = sum([isinstance(x, torch.Tensor) for x in args])
        if num_tensor_args == 1:
            check_schema("self: jt, ...", func, *args, **kwargs)
            return functools.partial(jagged_unary_pointwise, func)
        elif num_tensor_args == 2:
            check_schema("lhs: any, rhs: any", func, *args, **kwargs)
            return functools.partial(jagged_binary_pointwise, func)

    return None


def extract_kwargs(arg):
    kwargs = {
        "offsets": arg.offsets(),
        "_metadata_cache": arg._metadata_cache,
    }
    return kwargs


def jagged_unary_pointwise(func, *args, **kwargs):
    return NestedTensor(
        func(args[0]._values, *args[1:], **kwargs), **extract_kwargs(args[0])
    )


def jagged_binary_pointwise(func, *args, **kwargs):
    a, b = args[0], args[1]
    assert isinstance(a, NestedTensor) or isinstance(b, NestedTensor)

    mismatch_error_msg = (
        "cannot call binary pointwise function {} with inputs of shapes {} and {}"
    )
    # a is NT, b is NT
    if isinstance(a, NestedTensor) and isinstance(b, NestedTensor):
        # ex: (B, j0, D) + (B, j0, D)
        # ex: (B, j0, D) + (B, j0, 1)
        if raggedness_matches(a, b._size):
            return NestedTensor(
                func(a._values, b._values, *args[2:], **kwargs), **extract_kwargs(a)
            )
        raise RuntimeError(mismatch_error_msg.format(func.__name__, a._size, b._size))
    # either a is NT or b is NT at this point
    a_is_nt = isinstance(a, NestedTensor)
    extracted_kwargs = extract_kwargs(a) if a_is_nt else extract_kwargs(b)

    # === Handle broadcasting across the batch / ragged dims ===

    # Easy case: take advantage of pre-existing broadcasting logic
    # ex: (B, j0, ?, ?) + (?) -> (B, j0, ?, ?)
    # ex: (B, j0, ?, ?) + (?, ?) -> (B, j0, ?, ?)
    # ex: (B, j0, ?, ?) + (1, 1, ?, ?) -> (B, j0, ?, ?)
    nt, t = (a, b) if a_is_nt else (b, a)
    # See Note: [ Squeezing leading ones ]
    if t.dim() > nt.dim():
        raise NotImplementedError("NYI: broadcasting NT with T with larger dim")
    t_squeezed = squeeze_leading_ones(t)
    if nt.dim() >= t_squeezed.dim() + 2:
        lhs, rhs = (nt._values, t_squeezed) if a_is_nt else (t_squeezed, nt._values)
        return NestedTensor(func(lhs, rhs, *args[2:], **kwargs), **extracted_kwargs)

    # Harder case: do manual broadcasting over unbound components
    # when NT dim == non-NT dim
    # ex: (B, j0, D_0, D_1) + (B, 1, D_0, D_1) -> (B, j0, D_0, D_1)
    if a.dim() == b.dim():
        # ex: (B, j0, D_0, D_1) + (1, 1, D_0, D_1) -> should
        # be (B, j0, D_0, D_1) but not yet supported
        if a.shape[0] != b.shape[0]:
            raise RuntimeError(
                mismatch_error_msg.format(func.__name__, a.shape, b.shape)
            )

        # need to use offsets to broadcast across ragged dim properly
        # NB: inefficient fallback here; Triton codegen can help this
        # TODO: Make this work with autograd
        outputs = []
        for a_comp, b_comp in zip(a.unbind(), b.unbind()):
            outputs.append(func(a_comp, b_comp, *args[2:], **kwargs))
        new_values = torch.cat(outputs, dim=0)
        return NestedTensor(new_values, **extracted_kwargs)

    # ex: (B, j0, D_0, D_1) + (A, B, 1, D_0, D_1) -> error because this breaks the invariant
    # that ragged dim is wrt left-most batch dim
    raise RuntimeError(mismatch_error_msg.format(func.__name__, a.shape, b.shape))


def jagged_torch_function(func, *args, **kwargs):
    # SDPA has special kernels that handle nested tensors.
    # Dispatch to the correct implementation here
    if func is torch._C._nn.scaled_dot_product_attention:
        return jagged_scaled_dot_product_attention(*args, **kwargs)

    # Handle flatten() here because it's CompositeImplicit.
    if func.__name__ == "flatten":

        def _flatten_sig(input, start_dim=0, end_dim=-1):
            pass

        _, new_kwargs = normalize_function(
            _flatten_sig, args=args, kwargs=kwargs, normalize_to_only_use_kwargs=True
        )

        inp = new_kwargs.pop("input")
        new_kwargs["start_dim"] = _wrap_jagged_dim(
            inp.dim(), new_kwargs["start_dim"], "flatten"
        )
        new_kwargs["end_dim"] = _wrap_jagged_dim(
            inp.dim(), new_kwargs["end_dim"], "flatten"
        )

        return NestedTensor(func(inp._values, **new_kwargs), **extract_kwargs(inp))

    raise NotImplementedError(func)


@register_jagged_func(
    [
        torch.ops.aten.is_non_overlapping_and_dense.default,
        torch.ops.aten.sym_size.default,
        torch.ops.aten.dim.default,
        torch.ops.aten.sym_numel.default,
        torch.ops.aten.sym_stride.default,
        torch.ops.aten.sym_storage_offset.default,
    ],
    "self: jt_all",
)
def tensor_attr_supported_getter(func, *args, **kwargs):
    if func == torch.ops.aten.is_non_overlapping_and_dense.default:
        return False

    if func == torch.ops.aten.sym_size.default:
        return args[0]._size

    if func == torch.ops.aten.dim.default:
        return len(args[0]._size)

    if func == torch.ops.aten.sym_numel.default:
        if args[0]._lengths is not None:
            return int(sum(args[0]._lengths) * math.prod(args[0]._size[2:]))
        return args[0]._values.numel()

    if func == torch.ops.aten.sym_stride.default:
        return args[0]._strides

    if func == torch.ops.aten.sym_storage_offset.default:
        return args[0]._values.storage_offset()


@register_jagged_func(torch.ops.prim.layout.default, "self: jt_all")
def prim_layout_default(func, *args, **kwargs):
    return torch.jagged


@register_jagged_func(
    [torch.ops.aten.size.default],
    "self: jt_all",
)
def tensor_attr_unsupported_getter(func, *args, **kwargs):
    if func == torch.ops.aten.size.default:
        raise RuntimeError(
            "NestedTensors does not support directly calling torch.ops.aten.size "
            "please use `nested_tensor.size()` instead."
        )


@register_jagged_func(torch.ops.aten.is_contiguous.default, "self: jt_all")
def is_contiguous_general(func, *args, **kwargs):
    from torch._prims_common import is_contiguous_for_memory_format

    _, new_kwargs = normalize_function(
        func, args=args, kwargs=kwargs, normalize_to_only_use_kwargs=True
    )
    inp = new_kwargs.pop("input")

    # If created from narrow() check for lengths
    if inp.lengths() is not None:
        return False

    # If jagged dim is not 1 it's not contiguous
    if inp._ragged_idx != 1:
        return False

    new_kwargs["memory_format"] = new_kwargs.get(
        "memory_format", torch.contiguous_format
    )
    if new_kwargs["memory_format"] == torch.preserve_format:
        return True
    return is_contiguous_for_memory_format(inp.values(), **new_kwargs)


register_jagged_func(
    torch.ops.aten.is_contiguous.memory_format, "self: jt_all, memory_format: any?"
)(is_contiguous_general)


@register_jagged_func(torch.ops.aten.linear.default, "input: jt, weight: t, bias: t?")
def linear_default(func, *args, **kwargs):
    _, new_kwargs = normalize_function(
        func, args=args, kwargs=kwargs, normalize_to_only_use_kwargs=True
    )

    inp = new_kwargs.pop("input")

    return NestedTensor(func(inp._values, **new_kwargs), **extract_kwargs(inp))


@register_jagged_func(
    torch.ops.aten.linear_backward.default,
    "self: jt, grad_output: jt, weight: t, output_mask: any",
)
def linear_backward_default(func, *args, **kwargs):
    _, new_kwargs = normalize_function(
        func, args=args, kwargs=kwargs, normalize_to_only_use_kwargs=True
    )

    inp = new_kwargs.pop("input")
    grad_output = new_kwargs.pop("grad_output")
    weight = new_kwargs.pop("weight")

    check_ragged_dim_same(func, inp, "self", grad_output, "grad_output")
    ds = NestedTensor(
        torch.mm(grad_output._values, weight), **extract_kwargs(grad_output)
    )
    dw = torch.mm(grad_output._values.T, inp._values)
    db = None  # NYI: gradient for bias, need to reduce over ragged dim
    return (ds, dw, db)


@register_jagged_func(torch.ops.aten._to_copy.default, "self: jt")
def to_copy_default(func, *args, **kwargs):
    _, new_kwargs = normalize_function(
        func, args=args, kwargs=kwargs, normalize_to_only_use_kwargs=True
    )

    inp = new_kwargs.pop("input")
    # don't change layout
    new_kwargs.pop("layout")

    new_values = func(inp._values, **new_kwargs)
    # NB: Purposefully keep offsets on the old device.
    return NestedTensor(new_values, **extract_kwargs(inp))


register_jagged_func(
    [
        torch.ops.aten.ones_like.default,
        torch.ops.aten.zeros_like.default,
        torch.ops.aten.randn_like.default,
        torch.ops.aten.detach.default,
    ],
    "self: jt",
)(jagged_unary_pointwise)


register_jagged_func(
    torch.ops.aten._softmax.default, "self: jt, dim: any, half_to_float: any"
)(jagged_unary_pointwise)


@register_jagged_func(
    torch.ops.aten.native_dropout.default, "self: jt, float: any, train: any?"
)
def native_dropout_default(func, *args, **kwargs):
    _, new_kwargs = normalize_function(
        func, args=args, kwargs=kwargs, normalize_to_only_use_kwargs=True
    )

    inp = new_kwargs.pop("input")
    out1, out2 = func(inp._values, **new_kwargs)
    return (
        NestedTensor(out1, **extract_kwargs(inp)),
        NestedTensor(out2, **extract_kwargs(inp)),
    )


@register_jagged_func(
    torch.ops.aten.native_dropout_backward.default,
    "grad_output: jt, mask: jt, scale: any",
)
def native_dropout_backward_default(func, *args, **kwargs):
    _, new_kwargs = normalize_function(
        func, args=args, kwargs=kwargs, normalize_to_only_use_kwargs=True
    )
    grad_output = new_kwargs.pop("grad_output")
    mask = new_kwargs.pop("mask")
    return NestedTensor(
        func(grad_output._values, mask._values, **new_kwargs),
        **extract_kwargs(grad_output),
    )


@register_jagged_func(torch.ops.aten.prod.dim_int, "self: jt, dim: any, keepdim: any?")
def prod_dim_int(func, *args, **kwargs):
    _, new_kwargs = normalize_function(
        func, args=args, kwargs=kwargs, normalize_to_only_use_kwargs=True
    )

    inp = new_kwargs.pop("input")
    # TODO: Figure out how to handle this better
    # keep_dim is required to keep it in jagged format
    if not new_kwargs["keepdim"]:
        raise RuntimeError("prod(): keepdim=True must be set for NestedTensor")
    dim = new_kwargs["dim"]
    new_kwargs["dim"] = _wrap_jagged_dim(len(inp._size), dim, "prod")

    return NestedTensor(func(inp._values, **new_kwargs), **extract_kwargs(args[0]))


@register_jagged_func(
    torch.ops.aten.split.Tensor, "self: jt, split_size: any, dim: any"
)
def split_tensor(func, *args, **kwargs):
    _, new_kwargs = normalize_function(
        func, args=args, kwargs=kwargs, normalize_to_only_use_kwargs=True
    )

    inp = new_kwargs.pop("input")

    new_kwargs["dim"] = _wrap_jagged_dim(inp.dim(), new_kwargs["dim"], "split")

    return tuple(
        NestedTensor(values=x, **extract_kwargs(inp))
        for x in func(inp._values, **new_kwargs)
    )


@register_jagged_func(
    torch.ops.aten.split_with_sizes.default, "self: jt, split_sizes: any, dim: any"
)
def split_with_sizes_default(func, *args, **kwargs):
    _, new_kwargs = normalize_function(
        func, args=args, kwargs=kwargs, normalize_to_only_use_kwargs=True
    )

    inp = new_kwargs.pop("input")

    new_kwargs["dim"] = _wrap_jagged_dim(
        inp.dim(), new_kwargs["dim"], "split_with_sizes"
    )

    return [
        NestedTensor(values=x, **extract_kwargs(inp))
        for x in func(inp._values, **new_kwargs)
    ]


@register_jagged_func(torch.ops.aten.unbind.int, "self: jt_all, dim: any?")
def unbind_int(func, *args, **kwargs):
    # Note that this specializes on the length of the offsets
    _, new_kwargs = normalize_function(
        func, args=args, kwargs=kwargs, normalize_to_only_use_kwargs=True
    )

    dim = new_kwargs["dim"]
    if dim != 0:
        raise RuntimeError("unbind(): only supported for NestedTensor on dim=0")

    inp = new_kwargs.pop("input")
    values = inp.values()
    offsets = inp.offsets()
    lengths = inp.lengths()

    if inp._ragged_idx != 1:
        raise RuntimeError(
            "unbind(): only supported for NestedTensor when jagged dimension is 1"
        )

    if lengths is None:
        return torch.split(values, offsets.diff().tolist())
    return [
        values[offsets[i] : (offsets[i] + lengths[i])] for i in range(lengths.shape[0])
    ]


@register_jagged_func(torch.ops.aten.unsqueeze.default, "self: jt, dim: any")
def unsqueeze_default(func, *args, **kwargs):
    _, new_kwargs = normalize_function(
        func, args=args, kwargs=kwargs, normalize_to_only_use_kwargs=True
    )

    inp = new_kwargs.pop("input")
    values = inp._values

    # Account for collapsed jagged dim
    dim = new_kwargs["dim"]
    new_kwargs["dim"] = _wrap_jagged_dim(len(inp._size) + 1, dim, "unsqueeze")
    return NestedTensor(func(values, **new_kwargs), **extract_kwargs(inp))


@register_jagged_func(torch.ops.aten.cat.default, "tensors: any, dim: any")
def cat_default(func, *args, **kwargs):
    _, new_kwargs = normalize_function(
        func, args=args, kwargs=kwargs, normalize_to_only_use_kwargs=True
    )

    tensors = new_kwargs.pop("tensors")

    # Convert any non-nested to nested
    nested = [t for t in tensors if t.is_nested]
    assert len(nested) > 0
    first = nested[0]
    tensors = [t if t.is_nested else t.expand_as(first) for t in tensors]

    # Account for collapsed jagged dim
    dim = new_kwargs["dim"]
    new_kwargs["dim"] = _wrap_jagged_dim(len(first.shape), dim, "cat")

    return NestedTensor(
        func([t._values for t in tensors], **new_kwargs), **extract_kwargs(tensors[0])
    )


@register_jagged_func(torch.ops.aten.matmul.default, "self: jt, other: any")
def matmul_default(func, *args, **kwargs):
    _, new_kwargs = normalize_function(
        func, args=args, kwargs=kwargs, normalize_to_only_use_kwargs=True
    )

    inp = new_kwargs.pop("input")
    other = new_kwargs.pop("other")

    if inp.is_nested and not other.is_nested:
        return NestedTensor(
            func(inp._values, other, **new_kwargs), **extract_kwargs(inp)
        )
    elif inp.is_nested and other.is_nested:
        # BMM with equivalent ragged dims between the two inputs
        if inp.dim() > 3 and other.dim() > 3 and raggedness_matches(inp, other._size):
            return NestedTensor(func(inp._values, other._values), **extract_kwargs(inp))

    raise RuntimeError(
        f"matmul(): not supported between inputs of shapes {inp._size} and {other.shape}"
    )


@register_jagged_func(
    torch.ops.aten.expand.default, "self: jt, size: any, implicit: any?"
)
def expand_default(func, *args, **kwargs):
    _, new_kwargs = normalize_function(
        func, args=args, kwargs=kwargs, normalize_to_only_use_kwargs=True
    )

    inp = new_kwargs.pop("input")
    size = new_kwargs["size"]

    assert ("implicit" not in new_kwargs) or (not new_kwargs.pop("implicit"))
    if not raggedness_matches(inp, size):
        raise RuntimeError(f"expand(): cannot expand shape {inp._size} -> {size}")

    expand_arg = [-1, *size[2:]]
    return NestedTensor(func(inp._values, expand_arg), **extract_kwargs(inp))

# Allow both jt and t for self
@register_jagged_func(torch.ops.aten._nested_expand.default, "self: any, size: any, dummy: jt")
def _nested_expand(func, *args, **kwargs):
    _, new_kwargs = normalize_function(
        func, args=args, kwargs=kwargs, normalize_to_only_use_kwargs=True
    )
    inp = new_kwargs.pop("input")
    size = new_kwargs.pop("size")
    offsets, sum_offsets, *Ds = validate_and_get_meta("expand_as", size)
    if not inp.dim() <= len(size):
        fail_reason = "trying to expand input to a size with larger dim"
    if inp.is_nested:
        fail_reason = None
        if not inp.dim() == len(size):
            # We don't support cases like:
            # - (B, j0, D) -> (B, j0, D', D)
            fail_reason = "trying to expand input to a size with different dim"
        if not inp._ragged_idx == 1:
            # We could clean this up if we supported union types in the schema
            fail_reason = "transposed jagged layout nested tensor is not supported"
        if not size[1] == inp._size[1]:
            fail_reason = "input has a different raggedness than size"
        inp = inp._values
    else:
        # Currently you must either expand both B and j0 or neither, e.g.
        # one cannot expand (B, j0, D) to (B, 1, D) today.
        # Since j0 must be expanded for this branch, both are always expanded.
        # We only have 3 cases here:
        # - (1, 1, *Ds) -> (B, j0 ,*Ds)
        # - (1, *Ds) -> (B, j0, *Ds)
        # - (*Ds,) -> (B, j0, *Ds)
        # The case (1, 1, *Ds) is the only one we need to handle specially so that
        # it can be expanded (sum(*), *Ds)
        if inp.dim() == len(size):
            inp = inp.squeeze(0)
    if fail_reason is not None:
        raise ValueError(f"expand_as(): {fail_reason}")
    return NestedTensor(inp.expand([sum_offsets, *Ds]), offsets)


@register_jagged_func(torch.ops.aten.where.self, "condition: jt, self: jt, other: jt")
def where_self(func, *args, **kwargs):
    _, new_kwargs = normalize_function(
        func, args=args, kwargs=kwargs, normalize_to_only_use_kwargs=True
    )

    condition = new_kwargs.pop("condition")
    inp = new_kwargs.pop("input")
    other = new_kwargs.pop("other")

    assert condition._size == other._size == inp._size

    return NestedTensor(
        func(condition._values, inp._values, other._values, **new_kwargs),
        **extract_kwargs(condition),
    )


@register_jagged_func(torch.ops.aten._pin_memory.default, "self: jt, device: any?")
def _pin_memory_default(func, *args, **kwargs):
    _, new_kwargs = normalize_function(
        func, args=args, kwargs=kwargs, normalize_to_only_use_kwargs=True
    )

    inp = new_kwargs.pop("input")

    return NestedTensor(func(inp._values, **new_kwargs), **extract_kwargs(inp))


@register_jagged_func(torch.ops.aten.is_pinned.default, "self: jt, device: any?")
def is_pinned_default(func, *args, **kwargs):
    _, new_kwargs = normalize_function(
        func, args=args, kwargs=kwargs, normalize_to_only_use_kwargs=True
    )

    inp = new_kwargs.pop("input")

    return func(inp._values, **new_kwargs)


@register_jagged_func(torch.ops.aten.is_same_size.default, "self: jt, other: jt")
def is_same_size_default(func, *args, **kwargs):
    return args[0]._size == args[1]._size


@register_jagged_func(
    torch.ops.aten.sum.dim_IntList, "self: jt, dim: any?, keepdim: any?, dtype: any?"
)
def sum_dim_IntList(func, *args, **kwargs):
    # sum_dim_IntList can produce a NT or a T depending on whether the ragged dims
    # are reduced away.
    _, new_kwargs = normalize_function(
        func, args=args, kwargs=kwargs, normalize_to_only_use_kwargs=True
    )
    inp = new_kwargs.pop("input")
    assert inp._ragged_idx == 1
    new_kwargs["dim"], ragged_reduced_away = _wrap_jagged_dims(
        inp.dim(), new_kwargs["dim"], "sum"
    )

    if not ragged_reduced_away:
        return NestedTensor(func(inp._values, **new_kwargs), **extract_kwargs(inp))
    else:
        # Don't wrap because we reduced away the raggedness
        out = func(inp._values, **new_kwargs)
        if new_kwargs["keepdim"]:
            out = out.unsqueeze(0)
        return out

@register_jagged_func(
    torch.ops.aten.sum.default, "self: jt, dtype: any?"
)
def sum_default(func, *args, **kwargs):
    return args[0]._values.sum(**kwargs)


@register_jagged_func(
    torch.ops.aten.transpose.int, "self: jt_all, dim0: any, dim1: any"
)
def transpose_int(func, *args, **kwargs):
    _, new_kwargs = normalize_function(
        func, args=args, kwargs=kwargs, normalize_to_only_use_kwargs=True
    )

    from torch._prims_common import canonicalize_dims

    inp = new_kwargs.pop("input")
    dim0, dim1 = canonicalize_dims(inp.dim(), (new_kwargs["dim0"], new_kwargs["dim1"]))

    if inp._lengths is not None:
        raise ValueError(
            "transpose(): not supported on jagged layout nested tensor with holes"
        )

    # To support the SDPA API, inputs need to have the ragged idx transposed to dim 2
    # instead of 1, although the internal Flash and mem-effn implementations will
    # use the inputs with raggedness in dim 1.
    if dim0 == inp._ragged_idx or dim1 == inp._ragged_idx:
        if dim0 == 0 or dim1 == 0:
            raise ValueError(
                "Transpose is not supported on the batch dimension for jagged NT"
            )
        if dim0 == inp._ragged_idx:
            to_dim = dim1
        else:
            to_dim = dim0
        return NestedTensor(
            inp.values().transpose(
                _outer_to_inner_dim(len(inp._size), dim0),
                _outer_to_inner_dim(len(inp._size), dim1),
            ),
            **extract_kwargs(inp),
            _ragged_idx=to_dim,
        )

    new_kwargs["dim0"] = _wrap_jagged_dim(inp.dim(), new_kwargs["dim0"], "transpose")
    new_kwargs["dim1"] = _wrap_jagged_dim(inp.dim(), new_kwargs["dim1"], "transpose")

    return NestedTensor(func(inp._values, **new_kwargs), **extract_kwargs(inp))


@register_jagged_func(torch.ops.aten.view.default, "self: jt, size: any")
def view_default(func, *args, **kwargs):
    _, new_kwargs = normalize_function(
        func, args=args, kwargs=kwargs, normalize_to_only_use_kwargs=True
    )

    inp = new_kwargs.pop("input")
    size = new_kwargs.pop("size")

    # Ensure specified size still includes batch and ragged dims
    if len(size) < 3 or not raggedness_matches(inp, size):
        raise RuntimeError(f"view(): cannot view shape {inp._size} as {size}")

    jagged_size = [inp._values.shape[0]] + size[2:]
    return NestedTensor(func(inp._values, jagged_size), **extract_kwargs(inp))


@register_jagged_func(
    torch.ops.aten.native_layer_norm.default,
    "input: jt, normalized_shape: any, weight: any?, bias: any?, eps: any",
)
def native_layer_norm_default(func, *args, **kwargs):
    _, new_kwargs = normalize_function(
        func, args=args, kwargs=kwargs, normalize_to_only_use_kwargs=True
    )

    inp = new_kwargs.pop("input")
    normalized_shape = new_kwargs["normalized_shape"]

    # Ensure we're not trying to normalize over the ragged dim
    if inp.dim() < 3 or (inp.dim() - len(normalized_shape)) < 2:
        raise RuntimeError(
            "layer_norm(): normalizing over ragged dim not supported for nested tensors"
        )

    output, mean, std = func(inp._values, **new_kwargs)
    return (NestedTensor(output, **extract_kwargs(inp)), mean, std)


@register_jagged_func(
    torch.ops.aten.native_layer_norm_backward.default,
    "grad_out: jt, input: jt, normalized_shape: any, mean: any, rstd: any, weight: any?, bias: any?, output_mask: any",
)
def native_layer_norm_backward_default(func, *args, **kwargs):
    _, new_kwargs = normalize_function(
        func, args=args, kwargs=kwargs, normalize_to_only_use_kwargs=True
    )
    grad_out = new_kwargs.pop("grad_out")
    inp = new_kwargs.pop("input")
    d_input, d_gamma, d_beta = func(grad_out._values, inp._values, **new_kwargs)
    if d_input is None:
        return (None, d_gamma, d_beta)

    return (NestedTensor(d_input, **extract_kwargs(inp)), d_gamma, d_beta)


@register_jagged_func(torch.ops.aten.select.int, "self: jt, dim: any, index: any")
def select_int(func, *args, **kwargs):
    _, new_kwargs = normalize_function(
        func, args=args, kwargs=kwargs, normalize_to_only_use_kwargs=True
    )

    inp = new_kwargs.pop("input")
    new_kwargs["dim"] = _wrap_jagged_dim(inp.dim(), new_kwargs["dim"], "select")

    return NestedTensor(func(inp._values, **new_kwargs), **extract_kwargs(inp))


@register_jagged_func(
    torch.ops.aten.slice.Tensor,
    "self: jt, dim: any?, start: any?, end: any?, step: any?",
)
def slice_tensor(func, *args, **kwargs):
    _, new_kwargs = normalize_function(
        func, args=args, kwargs=kwargs, normalize_to_only_use_kwargs=True
    )

    inp = new_kwargs.pop("input")
    new_kwargs["dim"] = _wrap_jagged_dim(inp.dim(), new_kwargs["dim"], "slice")

    return NestedTensor(func(inp._values, **new_kwargs), **extract_kwargs(inp))


@register_jagged_func(
    torch.ops.aten.convolution.default,
    "input: jt, weight: t, bias: t?, stride: any, padding: any, "
    "dilation: any, transposed: any, output_padding: any, groups: any",
)
def convolution_default(func, *args, **kwargs):
    _, new_kwargs = normalize_function(
        func, args=args, kwargs=kwargs, normalize_to_only_use_kwargs=True
    )

    inp = new_kwargs.pop("input")

    return NestedTensor(func(inp._values, **new_kwargs), **extract_kwargs(inp))


@register_jagged_func(
    torch.ops.aten.mean.dim, "self: jt, dim: any?, keepdim: any, dtype: any?"
)
def mean_dim(func, *args, **kwargs):
    _, new_kwargs = normalize_function(
        func, args=args, kwargs=kwargs, normalize_to_only_use_kwargs=True
    )

    inp = new_kwargs.pop("input")
    # NB: mean expects dim as a single item list of ints for some reason
    new_kwargs["dim"] = [_wrap_jagged_dim(inp.dim(), new_kwargs["dim"][0], "mean")]

    return NestedTensor(func(inp._values, **new_kwargs), **extract_kwargs(inp))


@register_jagged_func(torch.ops.aten.stack.default, "tensors: any, dim: any")
def stack_default(func, *args, **kwargs):
    _, new_kwargs = normalize_function(
        func, args=args, kwargs=kwargs, normalize_to_only_use_kwargs=True
    )

    # guaranteed this is non-empty if we got here
    tensors = new_kwargs.pop("tensors")
    for t in tensors:
        if not isinstance(t, NestedTensor):
            raise RuntimeError("stack(): expected all nested tensors inputs")

        if t.dim() != tensors[0].dim():
            raise RuntimeError(
                "stack(): expected all nested tensors to have the same dim"
            )

        if not raggedness_matches(t, tensors[0].shape):
            raise RuntimeError(
                "stack(): expected all nested tensors to have the same nested structure"
            )

    new_kwargs["dim"] = _wrap_jagged_dim(
        tensors[0].dim() + 1, new_kwargs["dim"], "stack"
    )

    return NestedTensor(
        func([t._values for t in tensors], **new_kwargs), **extract_kwargs(tensors[0])
    )


@register_jagged_func(
    torch.ops.aten.embedding.default,
    "weight: t, indices: jt, padding_idx: any?, scale_grad_by_freq: any?, sparse: any?",
)
def embedding_default(func, *args, **kwargs):
    _, new_kwargs = normalize_function(
        func, args=args, kwargs=kwargs, normalize_to_only_use_kwargs=True
    )

    # guaranteed this is non-empty if we got here
    indices = new_kwargs.pop("indices")
    weight = new_kwargs.pop("weight")

    return NestedTensor(
        func(weight, indices._values, **new_kwargs), **extract_kwargs(indices)
    )


def get_factory_from_new_factory(aten_op):
    factory_map = {
        torch.ops.aten.new_zeros.default: torch.zeros,
        torch.ops.aten.new_empty.default: torch.empty,
        torch.ops.aten.new_full.default: torch.full,
        torch.ops.aten.new_ones.default: torch.ones,
    }
    return factory_map.get(aten_op, None)


# Note [ NestedTensor factory functions ]
#
# new_* functions are used to implement the factory functions for NestedTensor
# Here, `self` is only used to enable dispatching to happen.
# When someone calls into torch.zeros(sizes) where sizes contains a singleton,
# The torch.zeros(sizes) call will dispatch to nt.new_zeros(sizes)
def jagged_new_factory(func, *args, **kwargs):
    factory_fn = get_factory_from_new_factory(func)

    _, new_kwargs = normalize_function(
        func, args=args, kwargs=kwargs, normalize_to_only_use_kwargs=True
    )
    new_kwargs.pop("input")
    offsets, sum_offsets, *Ds = validate_and_get_meta(factory_fn.__name__, new_kwargs.pop("size"))
    return NestedTensor(factory_fn([sum_offsets, *Ds], **new_kwargs), offsets)


register_jagged_func(
    [
        torch.ops.aten.new_zeros.default,
        torch.ops.aten.new_empty.default,
        torch.ops.aten.new_full.default,
        torch.ops.aten.new_ones.default,
    ],
    "self: jt, size: any, dtype: any?, layout: any?, device: any?, pin_memory: any?",
)(jagged_new_factory)
