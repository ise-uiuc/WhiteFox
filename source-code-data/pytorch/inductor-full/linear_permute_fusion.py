def linear_permute_fusion(module: torch.fx.GraphModule) -> torch.fx.GraphModule:
    for node in module.graph.nodes:
        if (
            node.op == "call_method"
            and node.target == "permute"
            and check_permute(node)
        ):
            if len(node.args) > 0:
                input_node = node.args[0]
            else:
                input_node = node.kwargs["input"]
            if (
                input_node.op == "call_function"
                and input_node.target == torch.nn.functional.linear
            ):
                normalized = NormalizedLinearNode(input_node)
                input = normalized.get_input()
                weight = normalized.get_weight()
                bias = normalized.get_bias()
                with module.graph.inserting_before(node):
                    fused_node = module.graph.call_function(
                        linear_transpose, args=(input, weight, bias)
                    )
                    node.replace_all_uses_with(fused_node)
                    module.graph.erase_node(node)
                    if len(input_node.users) == 0:
                        module.graph.erase_node(input_node)

    module.graph.lint()
    module.recompile()
    return module
