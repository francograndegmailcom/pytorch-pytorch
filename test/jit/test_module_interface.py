# flake8: noqa
# TODO: enable linting check for this file

import torch
import torch.nn as nn
from jit_utils import JitTestCase

class OrigModule(nn.Module):
    def __init__(self):
        super(OrigModule, self).__init__()

    def one(self, inp1, inp2):
        # type: (Tensor, Tensor) -> Tensor
        return inp1 + inp2 + 1

    def two(self, input):
        # type: (Tensor) -> Tensor
        return input + 2

    def forward(self, input):
        # type: (Tensor) -> Tensor
        return input + self.one(input, input) + 1

class NewModule(nn.Module):
    def __init__(self):
        super(NewModule, self).__init__()

    def one(self, inp1, inp2):
        # type: (Tensor, Tensor) -> Tensor
        return inp1 * inp2 + 1

    def forward(self, input):
        # type: (Tensor) -> Tensor
        return self.one(input, input + 1)

class TestModuleInterface(JitTestCase):
    def test_not_submodule_interface_call(self):
        @torch.jit.interface
        class ModuleInterface(nn.Module):
            def one(self, inp1, inp2):
                # type: (Tensor, Tensor) -> Tensor
                pass

        class TestNotModuleInterfaceCall(nn.Module):
            proxy_mod : ModuleInterface

            def __init__(self):
                super(TestNotModuleInterfaceCall, self).__init__()
                self.proxy_mod = OrigModule()

            def forward(self, input):
                # type: (Tensor) -> Tensor
                return self.proxy_mod.two(input)

        with self.assertRaisesRegex(RuntimeError, "Tried to access nonexistent attribute"):
            torch.jit.script(TestNotModuleInterfaceCall())

    def test_module_swap(self):
        @torch.jit.interface
        class ModuleInterface(nn.Module):
            def one(self, inp1, inp2):
                # type: (Tensor, Tensor) -> Tensor
                pass

            def forward(self, input):
                # type: (Tensor) -> Tensor
                pass

        class TestModule(nn.Module):
            proxy_mod : ModuleInterface

            def __init__(self):
                super(TestModule, self).__init__()
                self.proxy_mod = OrigModule()

            def forward(self, input):
                # type: (Tensor) -> Tensor
                return self.proxy_mod.forward(input)

        scripted_mod = torch.jit.script(TestModule())
        input = torch.randn(3, 4)
        self.assertEqual(scripted_mod(input), 3 * input + 2)

        # module swap with module that have the same interface
        scripted_mod.proxy_mod = torch.jit.script(NewModule())
        self.assertEqual(scripted_mod(input), input * (input + 1) + 1)

        # module swap with non-scripted module should throw error
        with self.assertRaisesRegex(RuntimeError, "a ScriptModule with non-scripted module"):
            scripted_mod.proxy_mod = NewModule()

    def test_module_swap_wrong_module(self):
        @torch.jit.interface
        class ModuleInterface(nn.Module):
            def one(self, inp1, inp2):
                # type: (Tensor, Tensor) -> Tensor
                pass

            def forward(self, input):
                # type: (Tensor) -> Tensor
                pass

        class NewModuleWrong(nn.Module):
            def __init__(self):
                super(NewModuleWrong, self).__init__()

            def forward(self, input):
                # type: (int) -> int
                return input + 1

        class TestModule(nn.Module):
            proxy_mod : ModuleInterface

            def __init__(self):
                super(TestModule, self).__init__()
                self.proxy_mod = OrigModule()

            def forward(self, input):
                # type: (Tensor) -> Tensor
                return self.proxy_mod.forward(input)

        scripted_mod = torch.jit.script(TestModule())
        # module swap with in-compatible interface
        with self.assertRaisesRegex(RuntimeError, "is not compatible with interface"):
            scripted_mod.proxy_mod = torch.jit.script(NewModuleWrong())

    def test_module_swap_no_lazy_compile(self):
        @torch.jit.interface
        class ModuleInterface(nn.Module):
            def one(self, inp1, inp2):
                # type: (Tensor, Tensor) -> Tensor
                pass

            def forward(self, input):
                # type: (Tensor) -> Tensor
                pass

        class TestModule(nn.Module):
            proxy_mod : ModuleInterface

            def __init__(self):
                super(TestModule, self).__init__()
                self.proxy_mod = OrigModule()

            def forward(self, input):
                # type: (Tensor) -> Tensor
                return self.proxy_mod.forward(input)

        class NewModuleMethodNotLazyCompile(nn.Module):
            def __init__(self):
                super(NewModuleMethodNotLazyCompile, self).__init__()

            def one(self, inp1, inp2):
                # type: (Tensor, Tensor) -> Tensor
                return inp1 * inp2 + 1

            def forward(self, input):
                # type: (Tensor) -> Tensor
                return input + 1

        scripted_mod = torch.jit.script(TestModule())
        # module swap with module that have the same interface, but the method not get
        # lazily compiled from forward, user need to export it explicitly for swap to work
        with self.assertRaisesRegex(RuntimeError, "is not compatible with interface"):
            scripted_mod.proxy_mod = torch.jit.script(NewModuleMethodNotLazyCompile())

        class NewModuleMethodManualExport(nn.Module):
            def __init__(self):
                super(NewModuleMethodManualExport, self).__init__()

            @torch.jit.export
            def one(self, inp1, inp2):
                # type: (Tensor, Tensor) -> Tensor
                return inp1 * inp2 + 1

            def forward(self, input):
                # type: (Tensor) -> Tensor
                return input + 1

        scripted_mod.proxy_mod = torch.jit.script(NewModuleMethodManualExport())
        input = torch.randn(3, 4)
        self.assertEqual(scripted_mod(input), input + 1)

    def test_module_swap_no_module_interface(self):
        # test module swapping with no module interface
        class TestNoModuleInterface(nn.Module):
            def __init__(self):
                super(TestNoModuleInterface, self).__init__()
                self.proxy_mod = OrigModule()

            def forward(self, input):
                # type: (Tensor) -> Tensor
                return self.proxy_mod(input)

        scripted_no_module_interface = torch.jit.script(TestNoModuleInterface())
        # proxy mod is swapped with the new ScriptModule that share the same JIT type, should succeed.
        scripted_no_module_interface.proxy_mod = torch.jit.script(OrigModule())
        # proxy_mod is neither a module interface or have the same JIT type, should fail
        with self.assertRaisesRegex(RuntimeError,
                                    "Expected a value of type '__torch__.jit.test_module_interface.OrigModule' " +
                                    "for field 'proxy_mod', but found '__torch__.jit.test_module_interface.NewModule'"):
            scripted_no_module_interface.proxy_mod = torch.jit.script(NewModule())

    def test_script_module_as_interface_swap(self):
        @torch.jit.interface
        class ModuleInterface(nn.Module):
            def one(self, inp1, inp2):
                # type: (Tensor, Tensor) -> Tensor
                pass

            def forward(self, input):
                # type: (Tensor) -> Tensor
                pass

        class OrigScriptModule(torch.jit.ScriptModule):
            def __init__(self):
                super(OrigScriptModule, self).__init__()

            @torch.jit.script_method
            def one(self, inp1, inp2):
                # type: (Tensor, Tensor) -> Tensor
                return inp1 + inp2 + 1

            @torch.jit.script_method
            def forward(self, input):
                # type: (Tensor) -> Tensor
                return input + self.one(input, input) + 1

        class NewScriptModule(torch.jit.ScriptModule):
            def __init__(self):
                super(NewScriptModule, self).__init__()

            @torch.jit.script_method
            def one(self, inp1, inp2):
                # type: (Tensor, Tensor) -> Tensor
                return inp1 * inp2 + 1

            @torch.jit.script_method
            def forward(self, input):
                # type: (Tensor) -> Tensor
                return self.one(input, input + 1)

        class TestNNModuleWithScriptModule(nn.Module):
            proxy_mod : ModuleInterface

            def __init__(self):
                super(TestNNModuleWithScriptModule, self).__init__()
                self.proxy_mod = OrigScriptModule()

            def forward(self, input):
                # type: (Tensor) -> Tensor
                return self.proxy_mod.forward(input)

        input = torch.randn(3, 4)
        scripted_mod = torch.jit.script(TestNNModuleWithScriptModule())
        self.assertEqual(scripted_mod(input), 3 * input + 2)

        scripted_mod.proxy_mod = NewScriptModule()
        self.assertEqual(scripted_mod(input), input * (input + 1) + 1)
