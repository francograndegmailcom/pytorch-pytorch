import logging
import warnings
from typing import Any, Dict, Optional, Protocol, Tuple

import torch

from torch._library.utils import parse_namespace

log = logging.getLogger(__name__)


class FakeScriptObject:
    def __init__(self, wrapped_obj: Any, script_class_name: str):
        self.wrapped_obj = wrapped_obj

        # The fully qualified name of the class of original script object
        self.script_class_name = script_class_name


class HasStaticMethodFromReal(Protocol):
    @classmethod
    def from_real(cls, real_obj: torch.ScriptObject):
        pass


class FakeClassRegistry:
    def __init__(self):
        self._registered_class: Dict[str, Any] = {}

    def has_impl(self, full_qualname: str) -> bool:
        return full_qualname in self._registered_class

    def get_impl(self, full_qualname: str) -> Any:
        self._check_registered(full_qualname)
        return self._registered_class[full_qualname]

    def register(self, full_qualname: str, fake_class=None) -> None:
        if self.has_impl(full_qualname):
            warnings.warn(
                f"{full_qualname} is already registered. Previous fake class is overrided with {fake_class}."
            )
        self._registered_class[full_qualname] = fake_class

    def deregister(self, full_qualname: str) -> Any:
        if not self.has_impl(full_qualname):
            raise RuntimeError(
                f"Cannot deregister {full_qualname}. Please use register_fake_class to register it first."
                f" Or do you dereigster it twice?"
            )
        self._check_registered(full_qualname)
        return self._registered_class.pop(full_qualname)

    def clear(self) -> None:
        self._registered_class.clear()

    def _check_registered(self, full_qualname: str) -> None:
        if full_qualname not in self._registered_class:
            raise RuntimeError(
                f"{full_qualname} is not registered. Please use register_fake_class to register it first."
            )


global_fake_class_registry = FakeClassRegistry()


def to_fake_obj(fake_mode, x: torch.ScriptObject) -> FakeScriptObject:
    fake_x = _fake_obj_from_real(fake_mode, x)

    def _call_torchbind(method_name):
        from torch._higher_order_ops.torchbind import call_torchbind

        def wrapped(self_, *args, **kwargs):
            return call_torchbind(self_, method_name, *args, **kwargs)

        return wrapped

    fake_x_wrapped = FakeScriptObject(fake_x, x._type().qualified_name())  # type: ignore[attr-defined]
    for name in x._method_names():  # type: ignore[attr-defined]
        attr = getattr(fake_x, name, None)
        if attr:
            if not callable(attr):
                raise RuntimeError(f"Expect {name} to be a callable but got {attr}.")

            setattr(
                fake_x_wrapped,
                name,
                _call_torchbind(name).__get__(fake_x_wrapped),
            )
        else:
            log.warning("fake object of %s doesn't implement method %s.", x, name)
    return fake_x_wrapped


def register_fake_class(qualname, fake_class: Optional[HasStaticMethodFromReal] = None):
    r"""Register a fake implementation for this class.

    It's in the same spirit of registering a fake implementation for
    an operator but with the difference that it
    associates a fake class with the original torch bind class (registered
    with torch::class_). In this way, torch.compile can handle them properly
    in components such as Dynamo and AOTAutograd.

    This API may be used as a decorator (see example). For the fake class, users
    are required to provide a from_real classmethod that takes a real object and
    returns an instance of the fake class. All tensors in the fake object should also
    be properly fakified with to_fake_tensor() in from_real.

    Examples:
        # For a custom class Foo defined in test_custom_class_registration.cpp:
        TORCH_LIBRARY(_TorchScriptTesting, m) {
          m.class_<TensorQueue>("_TensorQueue")
            .def(torch::init<at::Tensor>())
            .def("push", &TensorQueue::push)
            .def("pop", &TensorQueue::pop)
            .def("top", &TensorQueue::top)
            .def("size", &TensorQueue::size)
            .def("clone_queue", &TensorQueue::clone_queue)
            .def_pickle(
                // __getstate__
                [](const c10::intrusive_ptr<TensorQueue>& self)
                    -> c10::Dict<std::string, at::Tensor> {
                  return self->serialize();
                },
                // __setstate__
                [](c10::Dict<std::string, at::Tensor> data)
                    -> c10::intrusive_ptr<TensorQueue> {
                  return c10::make_intrusive<TensorQueue>(std::move(data));
                });
            };
        # We could register a fake class FakeTensorQueue in Python as follows:
        import torch

        @torch._library.register_fake_class("_TorchScriptTesting::_TensorQueue")
        class FakeTensorQueue:
            def __init__(self, q):
                self.queue = q

            @classmethod
            def from_real(cls, real_tq):
                ctx = torch.library.get_ctx()
                fake_queue = [ctx.to_fake_tensor(t) for t in real_tq.clone_queue()]
                return cls(fake_queue)

            def push(self, x):
                self.queue.append(x)

            def pop(self):
                return self.queue.pop(0)

            def size(self):
                return len(self.queue)

    """

    def inner(fake_class: HasStaticMethodFromReal):
        ns, name = parse_namespace(qualname)

        # This also checks whether the refered torch::class_ exists.
        torchbind_class = torch._C._get_custom_class_python_wrapper(ns, name)

        from_method = getattr(fake_class, _CONVERT_FROM_REAL_NAME, None)
        if not from_method:
            raise RuntimeError(f"{fake_class} doesn't define a classmethod from_real.")

        if not isinstance(fake_class.__dict__[_CONVERT_FROM_REAL_NAME], classmethod):
            raise RuntimeError(
                f"{_CONVERT_FROM_REAL_NAME} method is not a classmethod."
            )

        global_fake_class_registry.register(_full_qual_class_name(qualname), fake_class)
        return fake_class

    if fake_class is None:
        return inner
    return inner(fake_class)


def deregister_fake_class(qualname):
    return global_fake_class_registry.deregister(_full_qual_class_name(qualname))


def has_fake_class(full_qualname) -> bool:
    return global_fake_class_registry.has_impl(full_qualname)


def find_fake_class(full_qualname) -> Optional[Any]:
    if not has_fake_class(full_qualname):
        return None
    return global_fake_class_registry.get_impl(full_qualname)


def _full_qual_class_name(qualname: str) -> str:
    ns, name = parse_namespace(qualname)
    return "__torch__.torch.classes." + ns + "." + name


# Return the namespace and class name from fully qualified name.
def _ns_and_class_name(full_qualname: str) -> Tuple[str, str]:
    splits = full_qualname.split(".")
    assert len(splits) == 5
    _torch, torch_ns, classes, ns, class_name = splits
    return ns, class_name


def _find_fake_class_for_script_object(x: torch.ScriptObject) -> Any:
    full_qualname = x._type().qualified_name()  # type: ignore[attr-defined]
    ns, class_name = _ns_and_class_name(full_qualname)
    fake_class = find_fake_class(full_qualname)
    if fake_class is None:
        raise RuntimeError(
            f" ScriptObject's {full_qualname} haven't registered a fake class."
            f" Please use register_fake_class({ns}::{class_name}) to annotate a fake class for the script obj."
            f" Specifically, create a python class that implements a fake version for all the methods"
            f" that're used in the program and put annotated class in the program e.g. after loading the library."
            f" The fake methods can be written in the same way as a meta kernel for an operator but need to additionally"
            f" simulate the object's states. Be sure to add a {_CONVERT_FROM_REAL_NAME} classmethod"
            f" to enable creating a fake obj from a real one."
        )
    return fake_class


_CONVERT_FROM_REAL_NAME = "from_real"


def _fake_obj_from_real(fake_mode, x) -> Any:
    fake_class = _find_fake_class_for_script_object(x)

    from_real_method = getattr(fake_class, _CONVERT_FROM_REAL_NAME, None)
    if not from_real_method:
        raise RuntimeError(
            f"{fake_class} must define a classmethod {_CONVERT_FROM_REAL_NAME}"
            f" that converts the real object to the fake object."
        )

    # from_real defined by user need the ctx to fakify the tensor states.
    ctx = torch._library.abstract_impl.AbstractImplCtx(fake_mode, None)
    with torch._library.abstract_impl.set_ctx_getter(lambda: ctx):
        return fake_class.from_real(x)
