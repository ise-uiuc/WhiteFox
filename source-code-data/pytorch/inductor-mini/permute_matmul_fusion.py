def permute_matmul_fusion(module: torch.fx.GraphModule) -> torch.fx.GraphModule:
    for node in module.graph.nodes:
        if node.op == "call_function" and (
            node.target == torch.bmm or node.target == torch.matmul
        ):
            normalized = NormalizedMatmulNode(node)
            input_A_node = normalized.get_input()
            input_B_node = normalized.get_other()
            Atrans = Btrans = False
            if (
                input_A_node.op == "call_method"
                and input_A_node.target == "permute"
                and check_permute(input_A_node)
            ):
                Atrans = True

            if (
                input_B_node.op == "call_method"
                and input_B_node.target == "permute"
                and check_permute(input_B_node)
            ):
                Btrans = True

            if Atrans or Btrans:
                module.graph.erase_node(node)
    return module
