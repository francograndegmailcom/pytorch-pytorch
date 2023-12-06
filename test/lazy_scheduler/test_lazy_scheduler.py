"""
pytest -vs test/lazy_scheduler/test_lazy_scheduler.py::TestLazyScheduler::test_backward_simple_no_segment
pytest -vs test/lazy_scheduler/test_lazy_scheduler.py::TestLazyScheduler::test_split_module_based_on_segment_info
pytest -vs test/lazy_scheduler/test_lazy_scheduler.py::TestLazyScheduler::test_compile_fx_with_segment_info
"""

import torch
import torch.utils._pytree as pytree
from torch.testing._internal.common_utils import TestCase as TorchTestCase
from torch._dynamo import disable
import functools
import itertools
from typing import Optional, Dict, Callable, List
from torch._subclasses.fake_tensor import FakeTensorMode
from collections import defaultdict, OrderedDict
import weakref
import threading
from torch.utils._python_dispatch import return_and_correct_aliasing
from torch._dynamo.backends.common import aot_autograd

fake_mode = FakeTensorMode()

class AsyncTensor(torch.Tensor):
  """
  This is a subclass of Tensor that represents a "lazy tensor".
  This tensor will be materialized by calling any tensor methods on it.
  """
  def __new__(cls, fake_tensor, *args, **kwargs):
    shape = fake_tensor.shape
    tensor_ctor_kwargs = {}
    tensor_ctor_kwargs["strides"] = fake_tensor.stride()
    tensor_ctor_kwargs["storage_offset"] = fake_tensor.storage_offset()
    tensor_ctor_kwargs["device"] = fake_tensor.device
    tensor_ctor_kwargs["layout"] = fake_tensor.layout
    tensor_ctor_kwargs["requires_grad"] = fake_tensor.requires_grad
    tensor_ctor_kwargs["dtype"] = fake_tensor.dtype
    out = torch.Tensor._make_wrapper_subclass(cls, shape, **tensor_ctor_kwargs)
    return out

  def __init__(self, fake_tensor, materialized_tensor=None):
    super().__init__()
    self._materialized_tensor = materialized_tensor
    self._fake = fake_tensor
    self._handle = None

  def async_repr(self):
    return f"AsyncTensor({self._handle}, {self._fake})"

  def __repr__(self):
    # NOTE: `print(tensor)` goes through this
    if self._handle is not None:
      AsyncTensor.wait_until_materialized([self])
      return self._materialized_tensor.__repr__()
    else:
      return self.async_repr()

  def __format__(self, format_spec):
    # NOTE: `print(f"{tensor}")` goes through this
    AsyncTensor.wait_until_materialized([self])
    return self._materialized_tensor.__format__(format_spec)

  def handle(self):
    assert self._handle is not None
    handle = self._handle()
    assert handle is not None
    return handle

  def set_handle(self, handle):
    self._handle = weakref.ref(handle)

  # NOTE: Any PyTorch reads or mutations in eager region will go through __torch_dispatch__,
  # so we materialize the underlying tensor here and returns it.
  @classmethod
  def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
    # TODO: implement randn_like etc. method that doesn't require a materialized tensor as input
    # TODO: implement other new_X etc. similar to new_empty_strided
    # print(f"func: {func}")
    if func in [torch.ops.aten.ones_like.default]:
      shape = args[0].shape
      dtype = args[0].dtype
      device = args[0].device
      requires_grad = args[0].requires_grad
      return torch.ones(shape, dtype=dtype, device=device, requires_grad=requires_grad)
    elif func in [torch.ops.aten.new_empty_strided.default]:
      args_except_self_tensor = list(args)[1:]
      return torch.empty_strided(*args_except_self_tensor, **kwargs)
    else:
      # TODO: handle tuple / list / etc in args
      # TODO: handle kwargs
      assert kwargs is None or len(kwargs) == 0
      AsyncTensor.wait_until_materialized(args)
      args_materialized = pytree.tree_map_only(AsyncTensor, lambda x: x._materialized_tensor, args)
      # kwargs_materialized = {k: pytree.tree_map_only(AsyncTensor, lambda x: x._materialized_tensor, v) for k, v in kwargs.items()}
      # out = func(*args_materialized, **kwargs_materialized)
      out = func(*args_materialized)
      # NOTE: if we don't re-wrap the output with AsyncTensor, sometimes the output will still be re-wrapped as AsyncTensor
      # (by another unknown mechanism outside of this code) but lose all its AsyncTensor attributes like `_materialized_tensor`
      if isinstance(out, torch.Tensor) and not isinstance(out, AsyncTensor):
        out = AsyncTensor(fake_tensor=fake_mode.from_tensor(out), materialized_tensor=out)
      return out
      # return return_and_correct_aliasing(func, args, kwargs, out)

  def materialize_with_value(self, tensor):
    self._materialized_tensor = tensor

  @staticmethod
  def check_materialized(async_tensors):
    all_materialized = True
    for t in async_tensors:
      if isinstance(t, AsyncTensor) and t._materialized_tensor is None:
          all_materialized = False
          break
    return all_materialized

  @staticmethod
  def wait_until_materialized(async_tensors):
    for async_tensor in async_tensors:
      if not AsyncTensor.check_materialized([async_tensor]):
        # NOTE: recursively schedule the deps first
        AsyncTensor.wait_until_materialized([async_tensor.handle().args])
        async_tensor.handle().schedule()
        async_tensor.handle().wait_for_completion()


class AsyncFuncHandle:
  """
  We use this class to represent the function that needs to be scheduled.
  It also has methods for checking whether the function has been scheduled or completed.
  """
  _gm_to_handle_mapping: Dict[torch.fx.GraphModule, "AsyncFuncHandle"] = {}

  def __init__(self, compiled_fn, args, outs_async, scheduler):
    self.cuda_event = torch.cuda.Event()
    self.compiled_fn: Callable = compiled_fn
    self.args = args
    self.outs_async = outs_async
    self.outs = None
    self.is_going_to_be_scheduled = False
    self._scheduler = weakref.ref(scheduler)

  def schedule(self):
    # make sure to schedule only once
    if self.is_going_to_be_scheduled:
      return
    self.is_going_to_be_scheduled = True
    gm = self._scheduler()._handle_to_gm_map[self]
    AsyncTensor.wait_until_materialized(self.args)
    args_materialized = pytree.tree_map_only(AsyncTensor, lambda x: x._materialized_tensor, pytree.tree_map(lambda x: x.detach(), self.args))
    self.outs = self.compiled_fn(list(args_materialized))
    self.cuda_event.record()

  def wait_for_completion(self):
    self.cuda_event.synchronize()
    for out, out_async in zip(self.outs, self.outs_async):
      # Set the output AsyncTensor's underlying materialized tensor
      # to be the actual output tensor.
      out_async.materialize_with_value(out)

  def is_completed(self):
    return self.cuda_event.query()

  def scheduler(self):
    scheduler = self._scheduler()
    assert scheduler is not None
    return scheduler


# NOTE: this is only for threading outputs through multiple submodules when doing module splitting above AOTAutograd.
# This is different from LazySchedulerGraphModule where the decision to run is given to the LazyScheduler.
# TODO: consider whether we can merge LazilyCompiledModule and LazySchedulerGraphModule
class _LazilyCompiledModule(torch.nn.Module):
  def __init__(self, submod, compiler):
    super().__init__()
    self.submod = submod
    self.compiler = compiler
    self.compiled = False

  def __call__(self, *args):
    if not self.compiled:
      new_submod = self.compiler(self.submod, args)
      del self.submod
      self.submod = new_submod
      self.compiled = True
      self.compiler = None
    x = self.submod(*args)
    return x


def split_module_based_on_segment_info(gm: torch.fx.GraphModule):
  known_segments = []
  for node in gm.graph.nodes:
    if len(known_segments) == 0 or node.meta["segment"] != known_segments[-1]:
      known_segments.append(node.meta["segment"])

  def split_callback(node):
    return known_segments.index(node.meta["segment"])

  qualname_map = {}
  gm_after_split = torch.fx.passes.split_module.split_module(
    m=gm,
    root_m=None,
    split_callback=split_callback,
    qualname_map=qualname_map,
    keep_original_order=True,
  )
  return gm_after_split


class LazySchedulerGraphModule(torch.nn.Module):
  """
  This module wraps around a GraphModule.
  Its __call__ method doesn't execute the graph module immediately.
  Instead, it calls the scheduler's maybe_run method, which decides
  whether to run the graph module based on the schedule.
  """
  def __init__(self, scheduler, gm, compiled_fn):
    super().__init__()
    self.scheduler = scheduler
    self.gm = gm
    self.compiled_fn = compiled_fn

  def __call__(self, *args):
    assert self.compiled_fn is not None
    return self.scheduler.maybe_run(self.gm, self.compiled_fn, *args)


class LazyScheduler:
  """
  LazyScheduler is used to decide when to schedule the execution of a graph module (based on the schedule).
  """
  def __init__(self):
    self._gm_to_handle_map = OrderedDict()
    self._handle_to_gm_map = OrderedDict()

  def _compile_fx_with_segment_info(
    self,
    # NOTE: matches positional args in compile_fx signature
    gm: torch.fx.GraphModule,
    example_inputs: List[torch.Tensor],
    segment_assignment_fn=None,
    **kwargs,
  ):
    if segment_assignment_fn is not None:
      segment_assignment_fn(gm)
    # Assumes `gm` already has segment info in each of its nodes
    gm_after_split = split_module_based_on_segment_info(gm)
    for name, sub_gm in gm_after_split.named_children():
      lazy_sub_gm = _LazilyCompiledModule(
        self,
        sub_gm,
        functools.partial(torch._inductor.compile_fx.compile_fx, **kwargs)
      )
      setattr(gm_after_split, name, lazy_sub_gm)
    # Trigger compile_fx in all submodules
    return gm_after_split(example_inputs)

  def compile_fx(
    self,
    # NOTE: matches positional args in compile_fx signature
    gm: torch.fx.GraphModule,
    example_inputs: List[torch.Tensor],
    **kwargs,
  ):
    return _compile_fx_with_segment_info(gm, example_inputs, segment_assignment_fn=set_segment_info, **kwargs)

  def compile_fx_inner(
    self,
    # NOTE: assumes first arg is GraphModule in compile_fx_inner signature
    gm: torch.fx.GraphModule,
    *args,
    **kwargs,
  ):
    """
    Compiles a graph module using Inductor compile_fx_inner,
    and wraps the output compiled_fn in a LazySchedulerGraphModule to be called later.
    """
    assert isinstance(gm, torch.fx.GraphModule)
    compiled_fn = torch._inductor.compile_fx.compile_fx_inner(gm, *args, **kwargs)
    lazy_gm = LazySchedulerGraphModule(
      self,
      gm,
      compiled_fn,
    )
    return lazy_gm

  def maybe_run(self, gm, compiled_fn, *args):
    """
    Decides whether to run the graph module based on the schedule.

    Always immediately returns AsyncTensor as output, and the AsyncTensor will be populated
    when the graph module is eventually executed.
    """
    # Create the handle and the async tensors
    args_fake = []
    for arg in args:
      if isinstance(arg, AsyncTensor):
        args_fake.append(arg._fake)
      elif isinstance(arg, torch.Tensor):
        args_fake.append(fake_mode.from_tensor(arg))
    with fake_mode:
      outs_fake = gm(*args_fake)

    outs_async = tuple(AsyncTensor(fake_tensor=out_fake) for out_fake in outs_fake)
    if gm in self._gm_to_handle_map:
      cur_handle = self._gm_to_handle_map[gm]
    else:
      cur_handle = AsyncFuncHandle(compiled_fn, args=args, outs_async=outs_async, scheduler=self)
      self._gm_to_handle_map[gm] = cur_handle
      self._handle_to_gm_map[cur_handle] = gm
    for out_async in outs_async:
      out_async.set_handle(cur_handle)

    # NOTE: add more complex logic here (e.g. check against the schedule, etc.)
    # cur_handle.schedule()
    # cur_handle.wait_for_completion()
    return cur_handle.outs_async


class TestCase(TorchTestCase):
  def setUp(self):
    torch._dynamo.reset()
    super().setUp()

  def tearDown(self):
    super().tearDown()
    torch._dynamo.reset()


class TestModule(torch.nn.Module):
  def __init__(self):
    super().__init__()

  def func1(self, x, y):
    z = torch.matmul(x, y)
    return z

  def forward(self, x, y):
    z = self.func1(x, y)
    z = z * z
    return z


class TestLazyScheduler(TestCase):
  def _validate(self, fn, backend, *args, skip_check=False):
    cloned_args = []
    for arg in args:
      cloned_args.append(arg.clone().detach().requires_grad_(arg.requires_grad))

    # Eager, 1st iter
    torch.manual_seed(0)
    expected = fn(*args)
    expected.sum().backward()

    # Eager, 2nd iter
    torch.manual_seed(0)
    expected = fn(*args)
    expected.sum().backward()

    compiled_fn = torch.compile(fn, fullgraph=False, backend=backend)

    # Compiled, 1st iter
    torch.manual_seed(0)
    result = compiled_fn(*cloned_args)
    result.sum().backward()

    # Compiled, 2nd iter
    torch.manual_seed(0)
    result = compiled_fn(*cloned_args)
    result.sum().backward()

    if not skip_check:
      self.assertEqual(
        result,
        expected,
        msg="Output mismatch between torch.compile and eager versions",
      )
      for arg, cloned_arg in zip(args, cloned_args):
        self.assertEqual(
          arg.grad,
          cloned_arg.grad,
          msg="Gradient mismatch between torch.compile and eager versions",
        )

  def test_backward_simple_no_segment(self):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    m = TestModule()
    m = m.to(device)
    x = torch.randn(4, 4, requires_grad=True, device=device)
    y = torch.randn(4, 4, requires_grad=True, device=device)

    lazy_scheduler = LazyScheduler()
    self._validate(
      m,
      # TODO: change this to use the new "split-subgraph-above-AOTAutograd" design
      functools.partial(compile_fx, inner_compile=lazy_scheduler.compile_fx_inner),
      x,
      y,
    )

  def test_split_module_based_on_segment_info(self):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    m = TestModule()
    m = m.to(device)
    x = torch.randn(4, 4, requires_grad=True, device=device)
    y = torch.randn(4, 4, requires_grad=True, device=device)

    def check_num_submods(gm, *args):
      num_call_fns = len([node for node in gm.graph.nodes if node.op == "call_function"])
      for i, node in enumerate(gm.graph.nodes):
        if node.op == "placeholder":
          node.meta["segment"] = "placeholder"
        elif node.op == "output":
          node.meta["segment"] = "output"
        elif node.op == "call_function":
          # Tag each call_function node with its own unique segment index
          node.meta["segment"] = i
      gm_after_split = split_module_based_on_segment_info(gm)
      self.assertEqual(len(list(gm_after_split.named_children())), num_call_fns)
      return gm_after_split

    compiled_m = torch.compile(m, backend=aot_autograd(fw_compiler=check_num_submods, bw_compiler=check_num_submods))
    out = compiled_m(x, y)
    out.sum().backward()

  def test_compile_fx_with_segment_info(self):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    m = TestModule()
    m = m.to(device)
    x = torch.randn(4, 4, requires_grad=True, device=device)
    y = torch.randn(4, 4, requires_grad=True, device=device)

    num_call_fns = None

    def segment_assignment_fn(gm):
      num_call_fns = len([node for node in gm.graph.nodes if node.op == "call_function"])
      for i, node in enumerate(gm.graph.nodes):
        if node.op == "placeholder":
          node.meta["segment"] = "placeholder"
        elif node.op == "output":
          node.meta["segment"] = "output"
        elif node.op == "call_function":
          # Tag each call_function node with its own unique segment index
          node.meta["segment"] = i

    def check_num_submods(gm, *args):
      self.assertEqual(len(list(gm.named_children())), num_call_fns)
      return gm

    lazy_scheduler = LazyScheduler()
    compiled_m = torch.compile(
      m,
      backend=functools.partial(
        functools.partial(lazy_scheduler._compile_fx_with_segment_info, segment_assignment_fn=segment_assignment_fn),
        inner_compile=check_num_submods
      ),
      fullgraph=False
    )
    out = compiled_m(x, y)
    out.sum().backward()

  def DISABLED_test_segment_tagging(self):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    m = TestModule()
    m = m.to(device)
    x = torch.randn(4, 4, requires_grad=True, device=device)
    y = torch.randn(4, 4, requires_grad=True, device=device)

    # Eager, 1st iter
    actual_e = m(x, y)
    actual_e.sum().backward()
    # Eager, 2nd iter
    actual_e = m(x, y)
    actual_e.sum().backward()

    register_segment(m.func1, "segment1")

    lazy_scheduler = LazyScheduler()

    def compile_then_check_segment_info(*args, **kwargs):
      lazy_gm = lazy_scheduler.compile(*args, **kwargs)
      for node in lazy_gm.graph.nodes:
        print(f"node.meta: {node.meta}")

    lazy_scheduler = LazyScheduler()
    compiled_m_ls = torch.compile(
      m,
      backend=functools.partial(lazy_scheduler.compile_fx, inner_compile=compile_then_check_segment_info),
      fullgraph=False
    )

"""
TODO:
0. Check gradients equivalence for eager vs. compile in test_backward_simple_no_segment
1. Add segment registration logic (do subgraph splitting above AOTAutograd, overwrite compile_fx and call compile_fx inside), enable test_segment_tagging to check segment tagging is working
2. Add scheduling logic, add unit test to check it's working
"""

if __name__ == "__main__":
  from torch._dynamo.test_case import run_tests
  run_tests()
