def fuse_binary(gm: torch.fx.GraphModule):
    modules = dict(gm.named_modules())
    for node in gm.graph.nodes:
        if check_node_is_binary(node) and check_binary_op_kwargs_is_default(node):
            for node_kind, fuse_func in computation_op_binary_op_fusion_map.items():
                if not isinstance(node.args[0], torch.fx.Node) or not isinstance(
                    node.args[1], torch.fx.Node
                ):
                    continue
                if not binary_inputs_meta_is_same(node):
                    continue
                attr = binary_attr[node.target]
                index_list = supported_index_list[attr]
                for index_dict in index_list:
                    index_node = index_dict["index_computation"]
                    index_pointwise = index_dict["index_pointwise"]
                    if check_node_kind(node.args[index_node], modules, node_kind):
                        if len(node.args[index_node].users) > 1:
                            continue
                        if not computation_module_used_once(node.args[index_node], gm):
                            continue
                        computation_node = modules[node.args[index_node].target]
                        if computation_node.training:
                            continue

                        # TODO: support padding str input("valid", "same").
                        if type(computation_node) in [nn.Conv2d] and isinstance(
                            computation_node.padding, str
                        ):
                            continue
                        # only fuse for linear when the dtype is bf16
                        if type(computation_node) in [
                            nn.Linear
                        ] and not is_bfloat16_module(computation_node):
                            continue
                        replace_and_fuse_for_binary(
                            computation_node,
                            node,
                            fuse_func,
                            attr if attr != "iadd" else "add",
                            modules,
                            index_node,
                            index_pointwise,
                        )
                        # Make sure the fused node is post node of node's inputs nodes.
                        node.append(node.args[index_node])
                        gm.graph.erase_node(node)
                        break
    gm.graph.lint()
    gm.recompile()
    return gm

computation_op_binary_op_fusion_map = {
    nn.Conv2d: fused_conv_binary_eval,
    nn.Linear: fused_linear_binary_eval,
}

binary_attr = {
    torch.add: "add",  # node.op == "call_function"
    "add": "add",  # node.op == "call_method"
    "add_": "iadd",  # node.op == "call_method"
    operator.add: "add",  # node.op == "call_function"
    operator.iadd: "iadd",  # node.op == "call_function"
    torch.sub: "sub",  # node.op == "call_function"
    "sub": "sub",  # node.op == "call_method"
    "sub_": "sub",  # node.op == "call_method"
    operator.sub: "sub",  # node.op == "call_function"
    operator.isub: "sub",  # node.op == "call_function"
}

supported_index_list = {
    "add": [
        {"index_computation": 0, "index_pointwise": 1},
        {"index_computation": 1, "index_pointwise": 0},
    ],
    "iadd": [{"index_computation": 0, "index_pointwise": 1}],
    "sub": [{"index_computation": 0, "index_pointwise": 1}],
}

def check_node_kind(current_node, modules, node_kind):
    if current_node.target not in modules:
        return False
    if type(modules[current_node.target]) is not node_kind:
        return False
    return True