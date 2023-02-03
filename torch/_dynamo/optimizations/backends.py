import functools
import logging
import os
import tempfile

import torch

from ..backends.registry import register_backend

from .subgraph import SubGraph

log = logging.getLogger(__name__)


def create_backend(fn):
    """
    WARNING: We do not recommend using this for new backends.  This is
    primarily used to support legacy TorchScript-based backends.
    """

    @functools.wraps(fn)
    def inner(model, example_inputs=None, **kwargs):
        if model is None:
            return None

        if not isinstance(model, SubGraph):
            with tempfile.TemporaryDirectory() as tmp:
                return inner(SubGraph(model, example_inputs, tmp), **kwargs)
        else:
            assert example_inputs is None

        try:
            return fn(model, **kwargs)
        except KeyboardInterrupt:
            raise

    return register_backend(inner)


def onnxrt_common(subgraph, provider, onnx_filename=None):
    import numpy as np  # type: ignore[import]
    import onnxruntime  # type: ignore[import]

    _np_dtype = {
        torch.float16: np.float16,
        torch.float32: np.float32,
        torch.float64: np.float64,
        torch.uint8: np.uint8,
        torch.int8: np.int8,
        torch.int16: np.int16,
        torch.int32: np.int32,
        torch.int64: np.longlong,
        torch.bool: np.bool_,
    }

    assert provider in onnxruntime.get_available_providers()
    session = onnxruntime.InferenceSession(
        onnx_filename or subgraph.onnx_filename, providers=[provider]
    )
    input_names = subgraph.input_names
    output_names = subgraph.output_names
    create_outputs = subgraph.empty_outputs_factory()
    is_cpu = subgraph.is_cpu

    def _call(*initial_args):
        binding = session.io_binding()
        args = [a.contiguous() for a in initial_args]
        for name, value in zip(input_names, args):
            dev = value.device
            binding.bind_input(
                name,
                dev.type,
                dev.index or 0,
                _np_dtype[value.dtype],
                value.size(),
                value.data_ptr(),
            )
        outputs = create_outputs()
        for name, value in zip(output_names, outputs):
            dev = value.device
            binding.bind_output(
                name,
                dev.type,
                dev.index or 0,
                _np_dtype[value.dtype],
                value.size(),
                value.data_ptr(),
            )
        session.run_with_iobinding(binding)
        if is_cpu:
            binding.copy_outputs_to_cpu()
        return outputs

    return subgraph.wrap_returns(_call)


@create_backend
def onnxrt_cpu(subgraph):
    return onnxrt_common(subgraph, provider="CPUExecutionProvider")


@create_backend
def onnxrt_cuda(subgraph):
    return onnxrt_common(subgraph, provider="CUDAExecutionProvider")


@create_backend
def onnx2tensorrt(subgraph):
    if subgraph.will_tensorrt_barf():
        # TensorRT fails violently with an abort() on this
        return None

    return onnxrt_common(subgraph, provider="TensorrtExecutionProvider")


@create_backend
def onnxrt_cpu_numpy(subgraph, provider="CPUExecutionProvider"):
    """Alternate version that integrates via numpy"""
    import onnxruntime

    assert provider in onnxruntime.get_available_providers()
    ort_session = onnxruntime.InferenceSession(
        subgraph.onnx_filename, providers=[provider]
    )

    def to_numpy(x):
        try:
            return x.numpy()
        except RuntimeError:
            return x.detach().numpy()

    def _call(*args):
        res = ort_session.run(
            None, {f"i{i}": to_numpy(arg) for i, arg in enumerate(args)}
        )
        res = [torch.from_numpy(x) for x in res]
        return res

    return subgraph.wrap_returns(_call)


@create_backend
def onnxrt(subgraph):
    if subgraph.is_cuda:
        return onnxrt_cuda(subgraph)
    else:
        return onnxrt_cpu(subgraph)


def _raise_timeout(signum, frame):
    raise TimeoutError()


@create_backend
def fx2trt(subgraph, **kwargs):
    if subgraph.will_tensorrt_barf():
        # TensorRT fails violently with an abort() on this
        return None

    from torch_tensorrt.fx.fx2trt import (  # type: ignore[import]
        InputTensorSpec,
        TRTInterpreter,
    )
    from torch_tensorrt.fx.passes.lower_basic_pass import (  # type: ignore[import]
        transform_setitem,
    )
    from torch_tensorrt.fx.tools.trt_splitter import (  # type: ignore[import]
        TRTSplitter,
        TRTSplitterSetting,
    )
    from torch_tensorrt.fx.tracer.acc_tracer import acc_tracer  # type: ignore[import]
    from torch_tensorrt.fx.trt_module import TRTModule  # type: ignore[import]
    from torch_tensorrt.fx.utils import LowerPrecision  # type: ignore[import]

    try:
        model = subgraph.model
        inputs = subgraph.example_inputs
        # pass rewrite
        model = transform_setitem(model, inputs)
        acc_model = acc_tracer.trace(model, inputs)
        # Split out unsupported ops
        splitter_setting = TRTSplitterSetting()
        splitter_setting.use_implicit_batch_dim = False
        splitter = TRTSplitter(acc_model, inputs, settings=splitter_setting)
        splitter.node_support_preview()
        split_mod = splitter()
        num_piece = 0
        for name, _ in split_mod.named_children():
            print(f"graph is split into {name}")
            num_piece += 1

        # if the graph module is split into pieces larger than 8, we consider its perf
        # is not good and fall back to non-TRT
        if num_piece > 8:
            print(
                f"The graph module is split into {num_piece} which is large than the \
                threshold=8. Fall back to non-TRT module."
            )
            return None

        if "fp16_mode" in kwargs and kwargs["fp16_mode"]:
            precision = LowerPrecision.FP16
        else:
            precision = LowerPrecision.FP32

        def get_submod_inputs(mod, submod, inputs):
            acc_inputs = None

            def get_input(self, inputs):
                nonlocal acc_inputs
                acc_inputs = inputs

            handle = submod.register_forward_pre_hook(get_input)
            mod(*inputs)
            handle.remove()
            return acc_inputs

        for name, _ in split_mod.named_children():
            if "_run_on_acc" in name:
                submod = getattr(split_mod, name)
                # print("acc=",submod.code)
                # Get submodule inputs for fx2trt
                acc_inputs = get_submod_inputs(split_mod, submod, inputs)

                # fx2trt replacement
                interp = TRTInterpreter(
                    submod,
                    InputTensorSpec.from_tensors(acc_inputs),
                    explicit_batch_dimension=True,
                )
                r = interp.run(
                    max_workspace_size=20 << 30,
                    lower_precision=precision,
                    # profiling_verbosity=trt.ProfilingVerbosity.DETAILED, #For profile
                )
                # For profile
                # from fx2trt_oss.fx.tools.trt_profiler_sorted import profile_trt_module
                # profile_trt_module("", trt_mod, acc_inputs)
                trt_mod = TRTModule(*r)

                setattr(split_mod, name, trt_mod)
            else:
                submod = getattr(split_mod, name)
                # print("gpu=",submod.code)
        return subgraph.wrap_returns(split_mod)
    except Exception:
        log.exception("FX2TRT conversion error")
        return None


@create_backend
def torch2trt(subgraph):
    if subgraph.will_tensorrt_barf():
        # TensorRT fails violently with an abort() on this
        return None

    from torch2trt import torch2trt  # type: ignore[import]

    inputs = subgraph.example_inputs
    trt_mod = torch2trt(
        subgraph.model,
        inputs,
        max_batch_size=len(inputs[0]),
        strict_type_constraints=True,
    )
    return subgraph.wrap_returns(trt_mod)


@create_backend
def tensorrt(subgraph):
    if subgraph.will_tensorrt_barf():
        # TensorRT fails violently with an abort() on this
        return None

    model = onnx2tensorrt(subgraph)
    if model is None:
        model = torch2trt(subgraph)
    return model


@create_backend
def cudagraphs(subgraph):
    model = subgraph.model
    inputs = subgraph.example_inputs
    assert subgraph.is_cuda
    return subgraph.wrap_returns(cudagraphs_inner(model, inputs))


def cudagraphs_inner(model, inputs, copy_outputs=True):
    assert isinstance(inputs, (list, tuple))
    static_inputs = [torch.zeros_like(x) for x in inputs]

    # warmup
    torch.cuda.synchronize()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        model(*inputs)
    stream.synchronize()
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()

    # record
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        static_outputs = model(*static_inputs)
    if not isinstance(static_outputs, (list, tuple)):
        static_outputs = (static_outputs,)

    def run(*new_inputs):
        assert len(static_inputs) == len(new_inputs)
        for dst, src in zip(static_inputs, new_inputs):
            dst.copy_(src)
        graph.replay()
        if copy_outputs:
            return [x.clone() for x in static_outputs]
        else:
            return static_outputs

    return run


def tvm_compile(jit_mod, example_inputs, log_file=None, **kwargs):
    if jit_mod is None:
        return None
    try:
        return tvm_compile_inner(jit_mod, example_inputs, None, log_file, **kwargs)
    except Exception as e:
        if log_file and os.path.exists(log_file):
            os.unlink(log_file)
        if isinstance(e, KeyboardInterrupt):
            raise
        log.exception("tvm error")
        return None


@create_backend
def tvm(subgraph):
    return subgraph.wrap_returns(
        tvm_compile_inner(
            subgraph.scripted,
            subgraph.example_inputs,
            tuning_option=None,
            cuda=subgraph.is_cuda,
        )
    )


@create_backend
def ansor(subgraph):
    """
    WARNING: this backend takes hours or days to train and
    often produces a slower result than the default schedule.
    """
    return subgraph.wrap_returns(
        tvm_compile_inner(
            subgraph.scripted,
            subgraph.example_inputs,
            tuning_option="auto_scheduler",
            log_file=subgraph.filename("ansor"),
            cuda=subgraph.is_cuda,
        )
    )


@create_backend
def tvm_meta_schedule(subgraph):
    return subgraph.wrap_returns(
        tvm_compile_inner(
            subgraph.scripted,
            subgraph.example_inputs,
            tuning_option="meta_schedule",
            trials=20000,
            cuda=subgraph.is_cuda,
        )
    )


@functools.lru_cache(None)
def llvm_target():
    if "avx512" in open("/proc/cpuinfo").read():
        return "llvm -mcpu=skylake-avx512"
    return "llvm -mcpu=core-avx2"


def tvm_compile_inner(
    jit_mod, example_inputs, tuning_option=None, log_file=None, trials=20000, cuda=False
):
    try:
        import tvm  # type: ignore[import]
        from tvm import relay  # type: ignore[import]
        from tvm.contrib import graph_executor  # type: ignore[import]

        shape_list = [(f"inp_{idx}", i.shape) for idx, i in enumerate(example_inputs)]
        mod, params = relay.frontend.from_pytorch(jit_mod, shape_list)
        if cuda:
            dev = tvm.cuda(0)
            target = tvm.target.cuda()
        else:
            dev = tvm.cpu(0)
            target = tvm.target.Target(llvm_target())

        if tuning_option == "auto_scheduler":
            from tvm import auto_scheduler

            if log_file is None:
                log_file = tempfile.NamedTemporaryFile()
            if not os.path.exists(log_file):
                tasks, task_weights = auto_scheduler.extract_tasks(
                    mod["main"], params, target
                )
                for task in tasks:
                    print(task.compute_dag)
                else:
                    print("No tasks")
                if len(tasks) != 0:
                    tuner = auto_scheduler.TaskScheduler(tasks, task_weights)
                    if not os.path.exists(log_file):
                        assert trials > 0
                        tune_option = auto_scheduler.TuningOptions(
                            num_measure_trials=trials,
                            measure_callbacks=[auto_scheduler.RecordToFile(log_file)],
                            early_stopping=2000,
                        )
                        try:
                            tuner.tune(tune_option)
                        except Exception:
                            if os.path.exists(log_file):
                                os.unlink(log_file)
                            raise

            with auto_scheduler.ApplyHistoryBest(log_file):
                with tvm.transform.PassContext(
                    opt_level=3, config={"relay.backend.use_auto_scheduler": True}
                ):
                    lib = relay.build(mod, target=target, params=params)
        elif tuning_option == "meta_schedule":
            from os import path as osp

            from tvm import meta_schedule as ms

            with tempfile.TemporaryDirectory() as work_dir:
                if log_file is not None:
                    assert osp.isdir(
                        log_file
                    ), "TVM's meta_schedule requires a directory for storing log files."
                    work_dir = log_file
                if not cuda:
                    # meta_schedule needs num-cores to be specified
                    # here we use the maximum core count
                    target = tvm.target.Target(
                        f"{llvm_target()} --num-cores {ms.utils.cpu_count(logical=False)}"
                    )
                # TODO(shingjan): This could be replaced by tvm.contrib.torch.optimize_torch
                # once USE_PT_TVMDSOOP is updated and turned on by default in TVM.
                database = ms.relay_integration.tune_relay(
                    mod=mod,
                    target=target,
                    work_dir=work_dir,
                    max_trials_global=20000,
                    num_trials_per_iter=64,
                    params=params,
                    strategy="evolutionary",
                )
                lib = ms.relay_integration.compile_relay(
                    database=database,
                    mod=mod,
                    target=target,
                    params=params,
                )

        elif tuning_option is None:
            # no autotuning (for debugging)
            with tvm.transform.PassContext(opt_level=10):
                lib = relay.build(mod, target=target, params=params)
        else:
            raise NotImplementedError(
                "This tuning option is invalid/not implemented for torchdynamo's TVM-related backend. "
                "There are three available options including None, auto_scheduler and meta_schedule."
            )
        m = graph_executor.GraphModule(lib["default"](dev))

        def to_torch_tensor(nd_tensor):
            """A helper function to transfer a NDArray to torch.tensor."""
            if nd_tensor.dtype == "bool":
                # DLPack does not support boolean so it can't be handled by
                # torch.utils.dlpack.from_pack. Workaround by going through
                # numpy, although this brings additional data copy overhead.
                return torch.from_numpy(nd_tensor.numpy())
            return torch.utils.dlpack.from_dlpack(nd_tensor.to_dlpack())

        def to_tvm_tensor(torch_tensor):
            """A helper function to transfer a torch.tensor to NDArray."""
            if torch_tensor.dtype == torch.bool:
                # same reason as above, fallback to numpy conversion which
                # could introduce data copy overhead
                return tvm.nd.array(torch_tensor.cpu().numpy())
            return tvm.nd.from_dlpack(torch_tensor)

        def exec_tvm(*i_args):
            args = [a.contiguous() for a in i_args]
            for idx, arg in enumerate(args, 0):
                if arg.dim() != 0:
                    if arg.requires_grad:
                        arg = arg.detach()
                    m.set_input(
                        f"inp_{idx}",
                        to_tvm_tensor(arg),
                    )
            m.run()
            return [
                to_torch_tensor(m.get_output(i)) for i in range(m.get_num_outputs())
            ]

        return exec_tvm
    except Exception:
        log.exception("tvm error")
        return jit_mod  # explicit fall back to eager


@create_backend
def ipex(subgraph):
    try:
        import intel_extension_for_pytorch  # type: ignore[import]  # noqa: F401
    except ImportError:
        log.exception(
            "Unable to import Intel Extension for PyTorch (IPEX). "
            "Please install the right version of IPEX that matches the PyTorch version being used. "
            "Refer to https://github.com/intel/intel-extension-for-pytorch for details."
        )
        raise

    from torch.utils._mode_utils import no_dispatch

    model = subgraph.model
    inputs = subgraph.example_inputs
    with no_dispatch():
        static_inputs = []
        for x in inputs:
            if x._has_symbolic_sizes_strides:
                size = [s.node.shape_env.size_hint(s.node.expr) for s in x.size()]
                stride = [s.node.shape_env.size_hint(s.node.expr) for s in x.stride()]
                static_inputs.append(
                    torch.as_strided(
                        torch.zeros(size, dtype=x.dtype, device=x.device), size, stride
                    )
                )
            else:
                static_inputs.append(torch.zeros_like(x))
    try:
        with torch.no_grad():
            traced_model = torch.jit.trace(model.eval(), static_inputs)
            traced_model = torch.jit.freeze(traced_model)
        return traced_model
    except Exception:
        log.warning("JIT trace failed during the 'ipex' optimize process.")
        return model


def fx2trt_compiler_fp16(gm: torch.fx.GraphModule, example_inputs):
    kwargs_fx2trt = {"fp16_mode": True}
    trt_compiled = fx2trt(gm, example_inputs, **kwargs_fx2trt)
    if trt_compiled is not None:
        return trt_compiled
    else:
        print(
            "FX2TRT conversion failed on the subgraph. Return GraphModule forward instead"
        )
        return gm.forward


def fx2trt_compiler(gm: torch.fx.GraphModule, example_inputs):
    kwargs_fx2trt = {"fp16_mode": False}
    trt_compiled = fx2trt(gm, example_inputs, **kwargs_fx2trt)
    if trt_compiled is not None:
        return trt_compiled
    else:
        print(
            "FX2TRT conversion failed on the subgraph. Return GraphModule forward instead"
        )
        return gm.forward
