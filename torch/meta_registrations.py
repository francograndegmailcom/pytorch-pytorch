import torch

Library = torch.library.Library

meta_lib = Library("aten", "IMPL", "Meta")

# Implementations below are taken from https://github.com/albanD/subclass_zoo/blob/main/python_meta_tensor.py
@torch.library.impl(meta_lib, "index_select")
def meta_index_select(self, dim, index):
    result_size = list(self.size())
    if self.dim() > 0:
        result_size[dim] = index.numel()
    return self.new_empty(result_size)

@torch.library.impl(meta_lib, "index_select.out")
def meta_index_select_out(self, out):
    torch._resize_output_(out, self.size(), self.device)
    return out.copy_(meta_index_select(self))

@torch.library.impl(meta_lib, "inverse")
def meta_inverse(self):
    if self.numel() == 0:
        return self.new_empty(self.size())
    inverse = self.new_empty(self.size())
    inverse.transpose_(-2, -1)
    return inverse

@torch.library.impl(meta_lib, "inverse.out")
def meta_inverse_out(self, out):
    torch._resize_output_(out, self.size(), self.device)
    return out.copy_(meta_inverse(self))

@torch.library.impl(meta_lib, "max")
def meta_max(self):
    return self.new_empty(())

@torch.library.impl(meta_lib, "max.out")
def meta_max_out(self, out):
    torch._resize_output_(out, self.size(), self.device)
    return out.copy_(meta_inverse(self))

@torch.library.impl(meta_lib, "abs")
def meta_abs(self):
    if self.is_complex():
        float_type = self.real.dtype
        return self.new_empty(self.size(), dtype=float_type)
    else:
        return self.new_empty(self.size())

@torch.library.impl(meta_lib, "abs.out")
def meta_abs_out(self, out):
    torch._resize_output_(out, self.size(), self.device)
    return out.copy_(meta_abs(self))

@torch.library.impl(meta_lib, "min")
def meta_min(self):
    return self.new_empty(())

@torch.library.impl(meta_lib, "min.out")
def meta_min_out(self, out):
    torch._resize_output_(out, self.size(), self.device)
    return out.copy_(meta_min(self))
