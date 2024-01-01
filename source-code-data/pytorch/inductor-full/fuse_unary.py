def fuse_unary(gm: torch.fx.GraphModule):
    modules = dict(gm.named_modules())

    for unary_op, (
        computation_module,
        fuse_func,
    ) in itertools.product(unary_ops, computation_op_unary_op_fusion_map.items()):
        pattern = (computation_module, unary_op)
        for node in gm.graph.nodes:
            if matches_module_pattern(
                pattern, node, modules
            ) or matches_module_function_pattern(pattern, node, modules):
                if (
                    len(node.args[0].users) > 1
                ):  # Output of computation_node is used by other nodes
                    continue
                if not computation_module_used_once(node.args[0], gm):
                    continue
                computation_node = modules[node.args[0].target]
                if node.op == "call_function" or node.op == "call_method":
                    # make sure unary function's inputs only one fx.node(others should be constant value).
                    if any(isinstance(v, torch.fx.Node) for v in node.args[1:]) or any(
                        isinstance(v, torch.fx.Node) for _, v in node.kwargs.items()
                    ):
                        continue
                    unary_node = create_unary_module(node)
                    unary_node.eval()
                else:
                    unary_node = modules[node.target]
                eval_mode = all(not n.training for n in [computation_node, unary_node])
                if not eval_mode:
                    continue
                # TODO: support padding str input("valid", "same").
                if type(computation_node) in [nn.Conv2d] and isinstance(
                    computation_node.padding, str
                ):
                    continue
                # TODO: support more conv+binary+unary fusion.
                if type(computation_node) in [ConvBinary2d] and type(
                    unary_node
                ) not in [nn.ReLU]:
                    continue
                # only fuse for linear when the dtype is bf16
                if type(computation_node) in [nn.Linear] and not is_bfloat16_module(
                    computation_node
                ):
                    continue
                # TODO: remove this when group depthwise ConvTranspose is supported
                if is_group_depthwise_conv_transpose(computation_node):
                    continue
                computation_node_input_size = (
                    node.args[0].args[0].meta.get("tensor_meta").shape
                )
                fused_module = fuse_func(
                    computation_node, unary_node, computation_node_input_size
                )
                replace_node_module(node.args[0], modules, fused_module)

                node.replace_all_uses_with(node.args[0])
                gm.graph.erase_node(node)
    gm.graph.lint()
    gm.recompile()
    return gm

unary_ops = [
    # modules
    nn.ReLU,
    nn.Sigmoid,
    nn.Tanh,
    nn.Hardswish,
    nn.LeakyReLU,
    nn.Hardtanh,
    nn.GELU,
    nn.ReLU6,
    nn.SiLU,
    nn.Hardsigmoid,
    # functional
    F.relu,
    F.sigmoid,
    F.tanh,
    F.hardswish,
    F.leaky_relu,
    F.hardtanh,
    F.gelu,
    F.relu6,
    F.silu,
    F.hardsigmoid,
    torch.relu,
    torch.sigmoid,
    torch.tanh,
    # methods (torch.Tensor.xxx)
    "relu",
    "sigmoid",
    "tanh",
]

computation_op_unary_op_fusion_map = {
    nn.Conv2d: fused_conv_unary_eval,
    nn.Linear: fused_linear_unary_eval,
    ConvBinary2d: fused_conv_binary_unary_eval,
    nn.ConvTranspose2d: fused_conv_transpose_unary_eval,
}
