import functools
from typing import List, Optional, Union

import torch
import torch._inductor.runtime.runtime_utils
from torch import Tensor
from torch._inductor import utils
from torch._subclasses.fake_tensor import FakeTensor
from torch.utils._mode_utils import no_dispatch
from ...utils._triton import has_triton

from ..pattern_matcher import fwd_only, gen_register_replacement, joint_fwd_bwd, Match

aten = torch.ops.aten


# This flag is only used for testing purpose.
# Changing it to True will ignore comparing do_bench times
# between original pattern and padded one.
_skip_do_bench_times = False


def fetch_fake_tensors(match, kwarg_names) -> List[Tensor]:
    kwargs = match.kwargs
    return [kwargs[name].meta["val"] for name in kwarg_names]


def unwrap_fake_args(*arg_names):
    def decorator(func):
        def wrapper(match):
            fake_tensors = fetch_fake_tensors(match, arg_names)
            return func(*fake_tensors)

        return wrapper

    return decorator


def get_alignment_size(x: Tensor) -> int:
    if x.dtype == torch.float16 or x.dtype == torch.half or x.dtype == torch.bfloat16:
        return 8
    elif x.dtype == torch.float32 or x.dtype == torch.float:
        return 4
    else:
        return 0


def check_device(a: Tensor, b: Tensor) -> bool:
    return a.is_cuda and b.is_cuda


def check_dtype(a: Tensor, b: Tensor) -> bool:
    return a.is_floating_point() and b.is_floating_point()


def should_pad_common(
    mat1: Tensor, mat2: Tensor, input: Optional[Tensor] = None
) -> bool:
    # It's fine we have symbolic shapes or strides as long as they
    # have hints. Later, we will make sure we only pad non-symbolic dimensions.
    def valid_shape_and_stride(t: Optional[Tensor]) -> bool:
        if t is None:
            return True

        symbolic_cnt = 0
        for x in t.size():
            if isinstance(x, int):
                continue
            elif utils.is_symbolic(x):
                if not x.node.has_hint():
                    return False
                symbolic_cnt += 1
            else:
                return False
        # filter out cases where all dimentions are symbolic
        if symbolic_cnt == len(t.size()):
            return False
        return all(
            isinstance(x, int) or (utils.is_symbolic(x) and x.node.has_hint())
            for x in t.stride()
        )

    return (
        torch._inductor.config.shape_padding
        and check_device(mat1, mat2)
        and check_dtype(mat1, mat2)
        and all(valid_shape_and_stride(t) for t in (mat1, mat2, input))
    )


def get_padded_length(x: Union[int, torch.SymInt], alignment_size) -> int:
    # we don't pad x if it is symbolic
    if isinstance(x, torch.SymInt) or alignment_size == 0 or x % alignment_size == 0:
        return 0
    return int((x // alignment_size + 1) * alignment_size) - x


def pad_dim(x: Tensor, padded_length: int, dim: int) -> Tensor:
    if padded_length == 0:
        return x
    pad = x.new_zeros(*x.shape[:dim], padded_length, *x.shape[dim + 1 :])
    return torch.cat([x, pad], dim=dim)


def addmm_pattern(
    input: Tensor, mat1: Tensor, mat2: Tensor, beta: float, alpha: float
) -> Tensor:
    return aten.addmm(input, mat1, mat2, beta=beta, alpha=alpha)


# addmm wrapper for testing purposes
def call_addmm(*args, **kwargs):
    return aten.addmm(*args, **kwargs)


# bmm wrapper for testing purposes
def call_bmm(*args, **kwargs):
    return aten.bmm(*args, **kwargs)


# mm wrapper for testing purposes
def call_mm(*args, **kwargs):
    return aten.mm(*args, **kwargs)


def should_pad_addmm(match: Match) -> bool:
    mat1, mat2, input = fetch_fake_tensors(match, ("mat1", "mat2", "input"))
    return should_pad_common(mat1, mat2, input) and should_pad_bench(
        match, mat1, mat2, torch.ops.aten.addmm, input=input
    )


def pad_addmm(
    input: Optional[Tensor],
    mat1: Tensor,
    mat2: Tensor,
    m_padded_length: int,
    k_padded_length: int,
    n_padded_length: int,
    beta=1.0,
    alpha=1.0,
    mat1_pre_padded: bool = False,
    mat2_pre_padded: bool = False,
):
    # for paddings, dim order is reversed for some reasons
    # and for every dim, we need to specify left and right padding
    mat1_padded = (
        mat1
        if mat1_pre_padded
        else pad_mat1(
            mat1, m_padded_length=m_padded_length, k_padded_length=k_padded_length
        )
    )
    mat2_padded = (
        mat2
        if mat2_pre_padded
        else pad_mat2(
            mat2, k_padded_length=k_padded_length, n_padded_length=n_padded_length
        )
    )
    if input is not None:
        if len(input.shape) < 2:
            # make sure we have at least two dimensions
            # the first one to be broadcasted over is sometimes implicit
            input = input.unsqueeze(0)
        if n_padded_length != 0 or m_padded_length != 0:
            bias_n_padded_length = n_padded_length
            bias_m_padded_length = m_padded_length
            # What if we're broadcasting?
            if input.shape[0] == 1 and mat1.shape[0] > 1:
                bias_m_padded_length = 0
            if input.shape[1] == 1 and mat2.shape[1] > 1:
                bias_n_padded_length = 0
            if bias_m_padded_length > 0 or bias_n_padded_length > 0:
                input_padded = aten.constant_pad_nd(
                    input, [0, bias_n_padded_length, 0, bias_m_padded_length]
                )
            else:
                input_padded = input
        else:
            input_padded = input
    else:
        input_padded = None

    try:
        res = call_addmm(input_padded, mat1_padded, mat2_padded, beta=beta, alpha=alpha)
    except RuntimeError as e:
        if input_padded is not None:
            note1 = f"\npad_addmm was called with argument shapes: input.shape={input.shape}, mat1.shape={mat1.shape}, mat2.shape={mat2.shape}, m_padded_length={m_padded_length}, k_padded_length={k_padded_length}, n_padded_length={n_padded_length}"  # type: ignore[union-attr] # noqa: B950
        else:
            note1 = f"pad_addmm was called with argument shapes: input_padded=None, mat1.shape={mat1.shape}, mat2.shape={mat2.shape}, m_padded_length={m_padded_length}, k_padded_length={k_padded_length}, n_padded_length={n_padded_length}"  # noqa: B950

        note2 = f"\naten.addmm was called with shapes: input_padded.shape={input_padded.shape}, mat1_padded.shape={mat1_padded.shape}, mat2_padded.shape={mat2_padded.shape}, beta={beta}, alpha={alpha}"  # noqa: B950
        raise RuntimeError(str(e) + note1 + note2) from e

    if m_padded_length != 0:
        res = res[:-m_padded_length, :]
    if n_padded_length != 0:
        res = res[:, :-n_padded_length]
    return res


def addmm_replace(
    input: Optional[Tensor], mat1: Tensor, mat2: Tensor, beta=1.0, alpha=1.0
) -> Tensor:
    k_padded_length = get_padded_length(mat1.shape[1], get_alignment_size(mat1))
    n_padded_length = get_padded_length(mat2.shape[1], get_alignment_size(mat2))
    m_padded_length = get_padded_length(mat1.shape[0], get_alignment_size(mat1))
    return pad_addmm(
        input,
        mat1,
        mat2,
        m_padded_length,
        k_padded_length,
        n_padded_length,
        beta,
        alpha,
    )


def is_mm_compute_bound(M: int, K: int, N: int, dtype: torch.dtype) -> bool:
    denominator = M * K + N * K + M * N
    if denominator == 0:
        return False
    arithmetic_intensity = (M * N * K) / denominator

    # Fails with AMD
    try:
        machine_balance = (
            1000 * utils.get_device_tflops(dtype)
        ) / utils.get_gpu_dram_gbps()
    except Exception:
        return True

    # dram_gbps might be underestimating bandwidth because of cache.
    # if we estimate machine balance too low we might miss some speedups,
    # if we extimate too high there will be unnecessary compilation time increase.
    # TODO - finetune coefficient here. As a reference point, Triton mm model assumes
    # 80% of reads are in cache and cache is 4x faster than dram_gbps
    machine_balance = machine_balance * 0.5

    return arithmetic_intensity > machine_balance


@functools.lru_cache(None)
def get_pad_cache():
    return torch._inductor.codecache.LocalCache()


def get_cached_should_pad(key):
    return get_pad_cache().lookup(key)


def set_cached_should_pad(key, value):
    return get_pad_cache().set_value(key, value=value)


def should_pad_bench_key(
    mat1: Tensor, mat2: Tensor, op, input: Optional[Tensor] = None
) -> str:
    def tensor_key(t):
        return (t.shape, t.stride(), t.dtype)

    tf32_key = (
        None if mat1.dtype != torch.float32 else torch.backends.cuda.matmul.allow_tf32
    )
    key = (
        tensor_key(mat1),
        tensor_key(mat2),
        op,
        input if input is None else tensor_key(input),
        tf32_key,
        torch._inductor.config.force_shape_pad,
    )

    return str(key)


def should_pad_bench(
    match, mat1: Tensor, mat2: Tensor, op, input: Optional[Tensor] = None
) -> bool:
    do_bench = functools.partial(
        torch._inductor.runtime.runtime_utils.do_bench,
        warmup=5,
    )
    m_padded_length = 0
    n_padded_length = 0
    batchsize = 1
    with no_dispatch():
        if op is torch.ops.aten.mm or op is torch.ops.aten.addmm:
            m = mat1.shape[0]
            k = mat1.shape[1]
            n = mat2.shape[1]
            k_padded_length = get_padded_length(k, get_alignment_size(mat1))
            n_padded_length = get_padded_length(n, get_alignment_size(mat2))
            m_padded_length = get_padded_length(m, get_alignment_size(mat1))
        elif op is torch.ops.aten.bmm:
            batchsize = mat1.shape[0]
            m = mat1.shape[1]
            k = mat1.shape[2]
            n = mat2.shape[2]
            k_padded_length = get_padded_length(k, get_alignment_size(mat1))
            m_padded_length = get_padded_length(m, get_alignment_size(mat1))
            n_padded_length = get_padded_length(n, get_alignment_size(mat2))
        else:
            return False

        if m_padded_length == k_padded_length == n_padded_length == 0:
            return False

        if torch._inductor.config.force_shape_pad:
            return True

        if not has_triton():
            return False

        if not is_mm_compute_bound(m, k, n, mat1.dtype):
            return False

        # We don't want to look up the cache for cases that are trivially false
        # since it does file io
        key = should_pad_bench_key(mat1, mat2, op, input)

        cached_pad = get_cached_should_pad(key)
        if cached_pad is not None:
            return cached_pad

        def realize_symbols(ds):
            return [d if isinstance(d, int) else d.node.hint for d in ds]

        def realize_tensor(t):
            if isinstance(t, FakeTensor):
                size_hints = realize_symbols(t.size())
                stride_hint = realize_symbols(t.stride())
                real_size = (
                    sum((d - 1) * s for d, s in zip(size_hints, stride_hint)) + 1
                )
                real_t = torch.randn(real_size, dtype=t.dtype, device=t.device)
                return torch.as_strided(real_t, size_hints, stride_hint)
            else:
                return torch.randn_like(t)

        mat1 = realize_tensor(mat1)
        mat2 = realize_tensor(mat2)
        if op is torch.ops.aten.bmm or op is torch.ops.aten.mm:
            ori_time = do_bench(
                lambda: op(mat1, mat2),
            )
        else:
            if input is not None:
                input = realize_tensor(input)
            ori_time = do_bench(
                lambda: op(input, mat1, mat2),
            )

        mat1_pad = mat1
        mat2_pad = mat2

        if op is torch.ops.aten.addmm:
            input_pad = None
            if input is not None and input.is_cuda:
                input_pad = torch.randn_like(input)
            pad_time = do_bench(
                lambda: pad_addmm(
                    input_pad,
                    mat1_pad,
                    mat2_pad,
                    m_padded_length,
                    k_padded_length,
                    n_padded_length,
                ),
            )
        elif op is torch.ops.aten.mm:
            pad_time = do_bench(
                lambda: pad_mm(
                    mat1_pad,
                    mat2_pad,
                    m_padded_length,
                    k_padded_length,
                    n_padded_length,
                ),
            )
        else:
            pad_time = do_bench(
                lambda: pad_bmm(
                    mat1_pad,
                    mat2_pad,
                    m_padded_length,
                    k_padded_length,
                    n_padded_length,
                ),
            )

        # Shape padding introduces additional memory ops. Based on microbenchmarks, 1.1x represents a reasonable
        # tradeoff between performance improvement from shape padding and overhead from additional memory ops
        # TODO: Build a learned model which would be better than this heuristic
        should_pad = _skip_do_bench_times or ori_time > pad_time * 1.1
        set_cached_should_pad(key, should_pad)

        return should_pad


def mm_pattern(mat1: Tensor, mat2: Tensor) -> Tensor:
    return aten.mm(mat1, mat2)


def should_pad_mm(match: Match) -> bool:
    mat1, mat2 = fetch_fake_tensors(match, ("mat1", "mat2"))
    return should_pad_common(mat1, mat2) and should_pad_bench(
        match, mat1, mat2, torch.ops.aten.mm
    )


def pad_mat1(mat1, *, m_padded_length, k_padded_length, is_bmm=False):
    if k_padded_length != 0 or m_padded_length != 0:
        # dim order is reversed for constant_pad_nd, for every dim we specify right and left padding
        pad_arg = [0, k_padded_length, 0, m_padded_length]
        if is_bmm:
            pad_arg.extend((0, 0))
        return aten.constant_pad_nd(mat1, pad_arg)
    return mat1


def pad_mat2(mat2, *,  k_padded_length, n_padded_length, is_bmm=False):
    if k_padded_length != 0 or n_padded_length != 0:
        # dim order is reversed for constant_pad_nd, for every dim we specify right and left padding
        pad_arg = [0, n_padded_length, 0, k_padded_length]
        if is_bmm:
            pad_arg.extend((0, 0))
        return aten.constant_pad_nd(mat2, pad_arg)
    else:
        return mat2


def pad_mm(
    mat1: Tensor,
    mat2: Tensor,
    m_padded_length: int,
    k_padded_length: int,
    n_padded_length: int,
    mat1_pre_padded: bool = False,
    mat2_pre_padded: bool = False,
) -> Tensor:
    mat1_padded = (
        mat1
        if mat1_pre_padded
        else pad_mat1(
            mat1, m_padded_length=m_padded_length, k_padded_length=k_padded_length
        )
    )
    mat2_padded = (
        mat2
        if mat2_pre_padded
        else pad_mat2(
            mat2, k_padded_length=k_padded_length, n_padded_length=n_padded_length
        )
    )
    res = call_mm(mat1_padded, mat2_padded)
    if m_padded_length != 0:
        res = res[:-m_padded_length, :]
    if n_padded_length != 0:
        res = res[:, :-n_padded_length]
    return res


def mm_replace(mat1: Tensor, mat2: Tensor) -> Tensor:
    k_padded_length = get_padded_length(mat1.shape[1], get_alignment_size(mat1))
    m_padded_length = get_padded_length(mat1.shape[0], get_alignment_size(mat1))
    n_padded_length = get_padded_length(mat2.shape[1], get_alignment_size(mat2))
    return pad_mm(
        mat1,
        mat2,
        m_padded_length,
        k_padded_length,
        n_padded_length,
    )


def bmm_pattern(mat1: Tensor, mat2: Tensor) -> Tensor:
    return aten.bmm(mat1, mat2)


def should_pad_bmm(match: Match) -> bool:
    mat1, mat2 = fetch_fake_tensors(match, ("mat1", "mat2"))
    return should_pad_common(mat1, mat2) and should_pad_bench(
        match, mat1, mat2, torch.ops.aten.bmm
    )


def pad_bmm(
    mat1: Tensor,
    mat2: Tensor,
    m_padded_length: int,
    k_padded_length: int,
    n_padded_length: int,
    mat1_pre_padded: bool = False,
    mat2_pre_padded: bool = False,
) -> Tensor:
    mat1_padded = (
        mat1
        if mat1_pre_padded
        else pad_mat1(
            mat1,
            m_padded_length=m_padded_length,
            k_padded_length=k_padded_length,
            is_bmm=True,
        )
    )
    mat2_padded = (
        mat2
        if mat2_pre_padded
        else pad_mat2(
            mat2,
            k_padded_length=k_padded_length,
            n_padded_length=n_padded_length,
            is_bmm=True,
        )
    )
    res = call_bmm(mat1_padded, mat2_padded)
    if m_padded_length != 0:
        res = res[:, :-m_padded_length, :]
    if n_padded_length != 0:
        res = res[:, :, :-n_padded_length]
    return res


def bmm_replace(mat1: Tensor, mat2: Tensor) -> Tensor:
    k_padded_length = get_padded_length(mat1.shape[2], get_alignment_size(mat1))
    n_padded_length = get_padded_length(mat2.shape[2], get_alignment_size(mat2))
    m_padded_length = get_padded_length(mat1.shape[1], get_alignment_size(mat1))
    return pad_bmm(
        mat1,
        mat2,
        m_padded_length,
        k_padded_length,
        n_padded_length,
    )


@functools.lru_cache(None)
def _pad_mm_init():
    from .joint_graph import patterns

    if torch.cuda.is_available():
        # workaround https://github.com/pytorch/pytorch/issues/97894
        device = "cuda"
    else:
        device = "cpu"

    # sizes/values dont actually matter for initial trace
    # once we get a possible match we re-trace with the actual values and verify the match still holds

    dim2a = functools.partial(torch.empty, (4, 4), device=device, requires_grad=True)
    dim2b = functools.partial(torch.empty, (4, 4), device=device, requires_grad=True)

    dim3a = functools.partial(torch.empty, (4, 4, 4), device=device, requires_grad=True)
    dim3b = functools.partial(torch.empty, (4, 4, 4), device=device, requires_grad=True)

    dim1a = functools.partial(torch.empty, (4), device=device, requires_grad=True)

    # workaround https://github.com/pytorch/pytorch/issues/97894
    # 0.113377 is a "magic" value that lets us recover the lost input arg relationship
    rep = {"beta": 0.213377, "alpha": 0.113377}

    for pattern, replacement, args, workaround, extra_check in [
        (
            mm_pattern,
            mm_replace,
            [dim2a(), dim2b()],
            {},
            should_pad_mm,
        ),
        (
            bmm_pattern,
            bmm_replace,
            [dim3a(), dim3b()],
            {},
            should_pad_bmm,
        ),
        (
            addmm_pattern,
            addmm_replace,
            [dim1a(), dim2a(), dim2b()],
            rep,
            should_pad_addmm,
        ),
    ]:
        assert isinstance(workaround, dict)  # mypy is unable to infer the type properly
        name = pattern.__name__

        gen_register_replacement(
            f"{name}_training",
            pattern,
            replacement,
            args,
            joint_fwd_bwd,
            patterns,
            extra_check=extra_check,
            scalar_workaround=workaround,
        )

        gen_register_replacement(
            f"{name}_inference",
            pattern,
            replacement,
            args,
            fwd_only,
            patterns,
            extra_check=extra_check,
            scalar_workaround=workaround,
        )
