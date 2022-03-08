import torch
import torch.fx as fx
import operator
import math
import torch.utils._pytree as pytree
import copy
import os
from torch.fx.passes import graph_drawer


class InvalidNodeBase(object):
    def __repr__(self):
        return "Invalid Node"


InvalidNode = InvalidNodeBase()


def _extract_graph_with_inputs_outputs(joint_graph, inputs, outputs):
    """
    Given a graph, extracts out a subgraph that takes the specified nodes as inputs and returns the specified outputs.

    This includes specifying non-placeholder nodes as inputs.

    The general strategy is to initialize all inputs with proxies as we
    encounter them, and trace through the graph, only keeping values which take
    in valid proxies. Then, all dead code is eliminated.
    """
    new_graph = fx.Graph()
    env = {}

    # Add new placeholder nodes in the order specified by the inputs
    for node in inputs:
        new_node = new_graph.placeholder(node.name)
        # Can't use node_copy here as we may be turning previous call_function into placeholders
        new_node.meta = node.meta
        env[node] = new_node

    for node in joint_graph.nodes:
        if node in inputs:
            continue
        elif node.op == 'placeholder':
            env[node] = InvalidNode
        elif node.op == 'call_function':
            all_args = pytree.tree_flatten((node.args, node.kwargs))[0]
            all_args = [isinstance(env[x], InvalidNodeBase) for x in all_args if isinstance(x, fx.Node)]
            if any(all_args):
                env[node] = InvalidNode
                continue
            env[node] = new_graph.node_copy(node, lambda x: env[x])
        elif node.op == 'get_attr':
            env[node] = new_graph.node_copy(node, lambda x: env[x])
        elif node.op == 'output':
            pass
    output_values = []
    for x in outputs:
        if isinstance(x, fx.Node):
            if x not in env:
                raise RuntimeError(f"Node {x} couldn't be found in env")
            output_values.append(env[x])
        else:
            output_values.append(x)
    new_graph.output(output_values)

    new_graph.eliminate_dead_code()
    new_graph.lint()
    return new_graph


def _is_primal(node):
    return node.op == "placeholder" and "tangents" not in node.target


def _is_tangent(node):
    return node.op == "placeholder" and "tangents" in node.target


def _extract_fwd_bwd_outputs(joint_module: fx.GraphModule):
    num_fwd_outputs = joint_module._out_spec.children_specs[0].num_leaves
    outputs = pytree.tree_flatten([node.args for node in joint_module.graph.nodes if node.op == 'output'])[0]
    fwd_outputs = outputs[:num_fwd_outputs]
    bwd_outputs = outputs[num_fwd_outputs:]
    return fwd_outputs, bwd_outputs


def _extract_fwd_bwd_modules(joint_module: fx.GraphModule, saved_values):
    fwd_outputs, bwd_outputs = _extract_fwd_bwd_outputs(joint_module)
    primal_inputs = list(filter(_is_primal, joint_module.graph.nodes))
    tangent_inputs = list(filter(_is_tangent, joint_module.graph.nodes))
    # Construct the forward module
    fwd_graph = _extract_graph_with_inputs_outputs(joint_module.graph, primal_inputs, fwd_outputs + saved_values)
    bwd_graph = _extract_graph_with_inputs_outputs(joint_module.graph, saved_values + tangent_inputs, bwd_outputs)

    # This is to filter out saved values that don't actually end up being used by the backwards pass
    for node in bwd_graph.nodes:
        if node.op == 'placeholder' and not node.users:
            for saved_value in saved_values:
                if saved_value.name == node.name:
                    saved_values.remove(saved_value)
                    break

    # Now, we re-generate the fwd/bwd graphs.
    # NB: This might increase compilation time, but I doubt it matters
    fwd_graph = _extract_graph_with_inputs_outputs(joint_module.graph, primal_inputs, fwd_outputs + saved_values)
    bwd_graph = _extract_graph_with_inputs_outputs(joint_module.graph, saved_values + tangent_inputs, bwd_outputs)

    fwd_module = fx.GraphModule(joint_module, fwd_graph)
    bwd_module = fx.GraphModule(joint_module, bwd_graph)
    return fwd_module, bwd_module


def default_partition(joint_module: fx.GraphModule, _joint_inputs):
    primal_inputs = list(filter(_is_primal, joint_module.graph.nodes))
    fwd_outputs, bwd_outputs = _extract_fwd_bwd_outputs(joint_module)
    forward_only_graph = _extract_graph_with_inputs_outputs(joint_module.graph, primal_inputs, fwd_outputs)
    forward_node_names = set([node.name for node in forward_only_graph.nodes if node.op != 'output'])

    def node_saved(node):
        return node.name in forward_node_names and 'tensor_meta' in node.meta
    saved_values = [node for node in joint_module.graph.nodes if node_saved(node)]
    return _extract_fwd_bwd_modules(joint_module, saved_values)


def _prod(x):
    s = 1
    for i in x:
        s *= i
    return s


def _size_of(metadata):
    sizes = {
        torch.float: 4,
        torch.float16: 2,
        torch.bfloat16: 2,
        torch.float32: 4,
        torch.float64: 8,
        torch.int: 4,
        torch.int8: 1,
        torch.int16: 2,
        torch.int32: 4,
        torch.int64: 8,
        torch.uint8: 1,
        torch.bool: 1,
    }

    numel = _prod(metadata.shape)
    dtype = metadata.dtype

    if dtype not in sizes:
        raise NotImplementedError("Don't know the size of dtype ", dtype)

    return numel * sizes[dtype]


def min_cut_rematerialization_partition(joint_module: fx.GraphModule, _joint_inputs):
    """
    Partitions the joint graph such that the backward recomputes the forward.
    Recomputing helps in trading off memory bandwidth with computation.

    To create the fwd and bwd graph, we copy the joint graph, manually set the
    outputs to just original forward or backward outputs. And then we run the
    resulting graphs through dead code elimintation.
    """
    try:
        import networkx as nx
    except ImportError:
        raise RuntimeError("Need networkx installed to perform smart recomputation heuristics")

    # draw_graph(joint_module, "joint.svg")
    full_bw_graph = joint_module.graph

    tangent_closure = set()
    name_to_node = {}
    for node in full_bw_graph.nodes:
        name_to_node[node.name] = node
        if node.op == 'placeholder' and "tangents" in node.target:
            tangent_closure.add(node)
        if node in tangent_closure:
            for user in node.users:
                tangent_closure.add(user)

    aten = torch.ops.aten

    pointwise_ops = [aten.add, aten.sub, aten.div, aten.atan2, aten.mul, aten.max, aten.min, aten.pow, aten.remainder, aten.fmod, aten.__and__, aten.__or__, aten.__xor__, aten.__lshift__, aten.__rshift__, aten.eq, aten.ne, aten.ge, aten.gt, aten.le, aten.lt, aten.abs, aten.bitwise_not, aten.ceil, aten.floor, aten.frac, aten.neg, aten.relu, aten.round, aten.silu, aten.trunc, aten.log, aten.log10, aten.log1p, aten.log2, aten.lgamma, aten.exp, aten.expm1, aten.erf, aten.erfc, aten.cos, aten.acos, aten.cosh, aten.sin, aten.asin, aten.sinh, aten.tan, aten.atan, aten.tanh, aten.atanh, aten.sqrt, aten.rsqrt,  aten.reciprocal, aten.sigmoid, aten.softplus, aten.threshold, aten.threshold_backward, aten.clamp, aten.where, aten.lerp, aten.addcmul, aten.gelu, aten.gelu_backward]  # noqa: E501
    misc_ops = [aten.to, aten.type_as, operator.getitem]

    # Ban reductions for now due to it being unnecessary/running into pathological situations
    # todo(chilli): add a heuristic to allow reduction only if output node is much smaller than input node
    reduction_ops = [aten.softmax, aten._softmax, aten._softmax_backward_data, aten.sum, aten.mean, aten._grad_sum_to_size, aten.sum_to_size, aten.amax]  # noqa: E501

    # not recomputed by default since these are kinda expensive/hard to fuse into
    # norm_ops = [aten.instance_norm, aten._batch_norm_impl_index, aten.native_batch_norm, aten.batch_norm, aten._batch_norm_impl_index_backward, aten.native_layer_norm, aten.layer_norm, aten.native_layer_norm_backward]  # noqa: E501

    # Not used by default since NVFuser can't fuse view ops
    # view_ops = [aten.expand, aten.clone, aten.transpose, aten.t, aten.view, aten._unsafe_view, aten.permute, aten.transpose, aten.t, aten._reshape_alias, aten.squeeze, aten.unsqueeze, aten.reshape, aten.cat, aten.slice, aten.split, aten.select, aten.repeat]  # noqa: E501

    unrecomputable_ops = [aten.mm, aten.convolution, aten.convolution_backward, aten.bmm, aten.addmm, aten.native_dropout, aten.rand_like, aten.randn_like, aten.upsample_bilinear2d]  # noqa: E501

    recomputable_ops = set(
        pointwise_ops
        + misc_ops
        + reduction_ops
        # + norm_ops
        # + view_ops
    )
    # ops = set([i.target for i in joint_module.graph.nodes if i.op == 'call_function'])
    # print(ops - recomputable_ops)
    AGGRESSIVE_RECOMPUTATION = False

    def ban_recomputation(node):
        if AGGRESSIVE_RECOMPUTATION:
            return (node.op == 'call_function' and node.target in unrecomputable_ops)
        else:
            if node.op != 'call_function':
                return False
            if node.target not in recomputable_ops:
                return True
            # If the output of the reduction is 4x smaller (arbitrary choice),
            # then we don't allow recomputation.
            if node.target in reduction_ops:
                input_tensors_size = sum([_size_of(i.meta['tensor_meta']) for i in node.args if isinstance(i, fx.Node)])
                output_size = _size_of(node.meta['tensor_meta'])
                return (output_size * 4 < input_tensors_size)
            return False

    def get_node_weight(node):
        mem_sz = _size_of(node.meta['tensor_meta'])
        if node.op == 'placeholder' and "primals" in node.target:
            return mem_sz
        else:
            return mem_sz * 2

    nx_graph = nx.DiGraph()
    for node in full_bw_graph.nodes:
        if node in tangent_closure and node.op != 'output':
            nx_graph.add_edge(node.name+"_in", "sink", capacity=math.inf)
            continue

        if node.op == 'placeholder' and "primals" in node.target:
            nx_graph.add_edge("source", node.name+"_in", capacity=math.inf)

        # If a node can't be recomputed (too expensive or involves randomness),
        # we prevent it from being recomputed by adding an inf edge to the source
        if ban_recomputation(node):
            nx_graph.add_edge("source", node.name+"_in", capacity=math.inf)

        if 'tensor_meta' not in node.meta:
            weight = math.inf
        else:
            weight = get_node_weight(node)

        # Creates the weights on the "node" edge
        nx_graph.add_edge(node.name+"_in", node.name+"_out", capacity=weight)
        for user in node.users:
            nx_graph.add_edge(node.name+"_out", user.name+"_in", capacity=math.inf)

    cut_value, partition = nx.minimum_cut(nx_graph, "source", "sink")
    reachable, non_reachable = partition
    cutset = set()
    for u, nbrs in ((n, nx_graph[n]) for n in reachable):
        cutset.update((u, v) for v in nbrs if v in non_reachable)

    cut_nodes = set()
    for node_in, node_out in cutset:
        assert node_in[:-3] == node_out[:-4]
        node_name = node_in[:-3]
        cut_nodes.add(node_name)
    # print(len(cut_nodes), sorted(list(cut_nodes)))

    saved_values = [name_to_node[node] for node in cut_nodes]

    return _extract_fwd_bwd_modules(joint_module, saved_values)


def draw_graph(traced: torch.fx.GraphModule, fname: str, figname: str = "fx_graph", clear_meta=True):
    if clear_meta:
        new_graph = copy.deepcopy(traced.graph)
        traced = fx.GraphModule(traced, new_graph)
        for node in traced.graph.nodes:
            node.meta = {}
    base, ext = os.path.splitext(fname)
    if not ext:
        ext = ".svg"
    print(f"Writing FX graph to file: {base}{ext}")
    g = graph_drawer.FxGraphDrawer(traced, figname)
    x = g.get_main_dot_graph()
    getattr(x, "write_" + ext.lstrip("."))(f"{base}{ext}")


def draw_joint_graph(graph, joint_inputs, file_name="full_graph.png"):
    draw_graph(graph, file_name)
    return default_partition(graph, joint_inputs)
