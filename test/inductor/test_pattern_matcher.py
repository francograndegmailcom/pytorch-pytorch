# Owner(s): ["module: inductor"]
import torch
from torch._dynamo.test_case import run_tests, TestCase
from torch._dynamo.utils import count_calls, counters
from torch._inductor.fx_passes import joint_graph
from torch._inductor.utils import run_and_get_code
from torch.testing._internal.common_utils import IS_LINUX
from torch.testing._internal.inductor_utils import HAS_CUDA


class TestPaternMatcher(TestCase):
    def test_mm_plus_mm(self):
        def fn(a, b, c, d):
            return torch.add(torch.mm(a, b), torch.mm(c, d))

        args_list = [
            (
                torch.randn(16, 16, device="cuda"),
                torch.randn(16, 16, device="cuda"),
                torch.randn(16, 16, device="cuda"),
                torch.randn(16, 16, device="cuda"),
            ),
            # https://github.com/pytorch/pytorch/issues/100670.
            (
                torch.randn(1, 4, device="cuda"),
                torch.randn(4, 2, device="cuda"),
                torch.randn(1, 2, device="cuda"),
                torch.randn(2, 1, device="cuda"),
            ),
            (
                torch.randn(1, 2, device="cuda"),
                torch.randn(2, 1, device="cuda"),
                torch.randn(1, 4, device="cuda"),
                torch.randn(4, 2, device="cuda"),
            ),
            (
                torch.randn(1, 4, device="cuda"),
                torch.randn(4, 2, device="cuda"),
                torch.randn(1, 5, device="cuda"),
                torch.randn(5, 2, device="cuda"),
            ),
        ]
        for args in args_list:
            counters.clear()
            expected = fn(*args)
            actual = torch.compile(fn)(*args)
            torch.testing.assert_close(actual, expected)
            self.assertEqual(counters["inductor"]["pattern_matcher_count"], 1)
            self.assertEqual(counters["inductor"]["pattern_matcher_nodes"], 3)

    def test_addmm(self):
        def fn(a, b, c):
            return torch.add(a, torch.mm(b, c)), torch.mm(b, c) + a

        args_list = [
            (
                torch.randn(16, 16, device="cuda"),
                torch.randn(16, 16, device="cuda"),
                torch.randn(16, 16, device="cuda"),
            ),
            (
                torch.randn(16, 16, device="cuda"),
                torch.randn(1, 16, device="cuda"),
                torch.randn(16, 16, device="cuda"),
            ),
            (
                torch.randn(1, 16, 16, device="cuda"),
                torch.randn(16, 16, device="cuda"),
                torch.randn(16, 16, device="cuda"),
            ),
            (4, torch.randn(16, 16, device="cuda"), torch.randn(16, 16, device="cuda")),
        ]
        for args in args_list:
            counters.clear()
            e1, e2 = fn(*args)
            a1, a2 = torch.compile(fn)(*args)
            torch.testing.assert_close(a1, e1)
            torch.testing.assert_close(a2, e2)
            self.assertEqual(counters["inductor"]["pattern_matcher_count"], 2)
            self.assertEqual(counters["inductor"]["pattern_matcher_nodes"], 4)

    def test_cat_mm(self):
        def fn(a, b, c):
            return torch.cat(
                [
                    torch.mm(a, b),
                    torch.mm(b, c),
                    torch.mm(a, c),
                ],
                1,
            )

        args = [
            torch.randn(16, 16, device="cuda"),
            torch.randn(16, 16, device="cuda"),
            torch.randn(16, 16, device="cuda"),
        ]
        expected = fn(*args)
        actual = torch.compile(fn)(*args)
        torch.testing.assert_close(actual, expected)
        self.assertEqual(counters["inductor"]["pattern_matcher_count"], 1)
        self.assertEqual(counters["inductor"]["pattern_matcher_nodes"], 4)

    def test_cat_addmm(self):
        def fn(a, b, c):
            return torch.cat(
                [
                    torch.addmm(a, b, c),
                    torch.addmm(b, c, a),
                    torch.addmm(c, a, b),
                ],
                1,
            )

        args = [
            torch.randn(16, 16, device="cuda"),
            torch.randn(16, 16, device="cuda"),
            torch.randn(16, 16, device="cuda"),
        ]
        expected = fn(*args)
        actual = torch.compile(fn)(*args)
        torch.testing.assert_close(actual, expected)
        self.assertEqual(counters["inductor"]["pattern_matcher_count"], 1)
        self.assertEqual(counters["inductor"]["pattern_matcher_nodes"], 4)

    def test_cat_slice_cat(self):
        def fn(a, b):
            cat_1 = torch.ops.aten.cat.default([a, b], 1)
            slice_1 = torch.ops.aten.slice.Tensor(cat_1, 0, 0, 9223372036854775807)
            slice_2 = torch.ops.aten.slice.Tensor(slice_1, 1, 0, 19)
            return torch.ops.aten.cat.default([cat_1, slice_2], 1)

        args = [
            torch.randn(2, 32, device="cuda"),
            torch.randn(2, 16, device="cuda"),
        ]
        expected = fn(*args)
        actual = torch.compile(fn)(*args)
        torch.testing.assert_close(actual, expected)
        self.assertEqual(counters["inductor"]["pattern_matcher_count"], 1)
        self.assertEqual(counters["inductor"]["pattern_matcher_nodes"], 4)

        counters.clear()
        args = [
            torch.randn(2, 8, device="cuda"),
            torch.randn(2, 16, device="cuda"),
        ]
        expected = fn(*args)
        actual = torch.compile(fn)(*args)
        torch.testing.assert_close(actual, expected)
        self.assertEqual(counters["inductor"]["pattern_matcher_count"], 1)
        self.assertEqual(counters["inductor"]["pattern_matcher_nodes"], 4)

        # Verify we fallback to non-optimal path for negative `end`.
        def fn(a, b):
            cat_1 = torch.ops.aten.cat.default([a, b], 1)
            slice_1 = torch.ops.aten.slice.Tensor(cat_1, 0, 0, 9223372036854775807)
            slice_2 = torch.ops.aten.slice.Tensor(slice_1, 1, 0, -1)
            return torch.ops.aten.cat.default([cat_1, slice_2], 1)

        counters.clear()
        args = [
            torch.randn(2, 8, device="cuda"),
            torch.randn(2, 16, device="cuda"),
        ]
        expected = fn(*args)
        actual = torch.compile(fn)(*args)
        torch.testing.assert_close(actual, expected)
        self.assertEqual(counters["inductor"]["pattern_matcher_count"], 1)
        self.assertEqual(counters["inductor"]["pattern_matcher_nodes"], 4)

    def test_pointless_convert(self):
        def fn1(x):
            x = torch.ops.prims.convert_element_type.default(x, torch.float16)
            x = torch.ops.prims.convert_element_type.default(x, torch.float32)
            return x

        gm = torch.fx.symbolic_trace(fn1)
        self.assertEqual(count_calls(gm.graph), 2)
        joint_graph.joint_graph_passes(gm)
        self.assertEqual(count_calls(gm.graph), 1)

        def fn2(x):
            x = torch.ops.prims.convert_element_type.default(x, torch.int32)
            x = torch.ops.prims.convert_element_type.default(x, torch.float32)
            return x

        gm = torch.fx.symbolic_trace(fn2)
        self.assertEqual(count_calls(gm.graph), 2)
        joint_graph.joint_graph_passes(gm)
        self.assertEqual(count_calls(gm.graph), 2)

    def test_pointless_cumsum(self):
        def fn1():
            ones = torch.full(
                [1, 128], 1, layout=torch.strided, dtype=torch.float32
            ).to(torch.int64)
            return torch.cumsum(ones, 1) * ones

        def fn2():
            ones = torch.full(
                [55, 10], 1, layout=torch.strided, dtype=torch.float32
            ).to(torch.int64)
            return torch.cumsum(ones, 1)

        for fn in (fn1, fn2):
            result, (code,) = run_and_get_code(torch.compile(fn, fullgraph=True))
            self.assertNotIn("aten.cumsum", code)
            self.assertEqual(result, fn())
            self.assertEqual(counters["inductor"]["pattern_matcher_count"], 1)
            counters.clear()

    def test_splitwithsizes_cat(self):
        # Good case
        def fn(a):
            split_with_sizes = torch.ops.aten.split_with_sizes.default(a, [8, 24], 1)
            getitem = split_with_sizes[0]
            getitem_1 = split_with_sizes[1]
            cat = torch.ops.aten.cat.default([getitem, getitem_1], 1)
            return cat**2

        args = [
            torch.randn(2, 32, device="cuda"),
        ]
        expected = fn(*args)
        actual = torch.compile(fn)(*args)
        torch.testing.assert_close(actual, expected)
        self.assertEqual(counters["inductor"]["pattern_matcher_count"], 1)
        self.assertEqual(counters["inductor"]["pattern_matcher_nodes"], 4)
        counters.clear()

        # Not all getitems are passed to cat
        def fn(a):
            split_with_sizes = torch.ops.aten.split_with_sizes.default(a, [8, 8, 16], 1)
            getitem = split_with_sizes[0]
            getitem_1 = split_with_sizes[1]
            getitem_2 = split_with_sizes[2]
            cat = torch.ops.aten.cat.default([getitem, getitem_1], 1)
            return cat**2 + getitem_2

        args = [
            torch.randn(2, 32, device="cuda"),
        ]
        expected = fn(*args)
        actual = torch.compile(fn)(*args)
        torch.testing.assert_close(actual, expected)
        self.assertEqual(counters["inductor"]["pattern_matcher_count"], 0)
        self.assertEqual(counters["inductor"]["pattern_matcher_nodes"], 0)
        counters.clear()

        # Different dimensions  (TODO this case should be handled by replacing with a reshape)
        def fn(a):
            split_with_sizes = torch.ops.aten.split_with_sizes.default(
                a, [8, 8, 8, 8], 1
            )
            cat = torch.ops.aten.cat.default(split_with_sizes, 0)
            return cat**2

        args = [
            torch.randn(2, 32, device="cuda"),
        ]
        expected = fn(*args)
        actual = torch.compile(fn)(*args)
        torch.testing.assert_close(actual, expected)
        self.assertEqual(counters["inductor"]["pattern_matcher_count"], 0)
        self.assertEqual(counters["inductor"]["pattern_matcher_nodes"], 0)

        # https://github.com/pytorch/pytorch/issues/99686.
        def fn(a):
            x = torch.ops.aten.split_with_sizes.default(a, [3, 2, 3], dim=1)
            cat = torch.ops.aten.cat.default([x[1], x[0], x[2]], dim=1)
            return cat

        args = [
            torch.randn(1, 8, device="cuda"),
        ]
        expected = fn(*args)
        actual = torch.compile(fn)(*args)
        torch.testing.assert_close(actual, expected)
        self.assertEqual(counters["inductor"]["pattern_matcher_count"], 0)
        self.assertEqual(counters["inductor"]["pattern_matcher_nodes"], 0)


if __name__ == "__main__":
    if IS_LINUX and HAS_CUDA:
        run_tests()
