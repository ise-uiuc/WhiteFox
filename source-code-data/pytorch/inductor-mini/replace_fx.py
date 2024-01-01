def replace_fx(gm: torch.fx.GraphModule, example_inputs):
    for node in reversed(list(gm.graph.nodes)):
        if node.op == "call_function" and node.target in replacements:
            if (
                config.fallback_random
                and replacements[node.target] in replacements_using_triton_random
            ) or (is_cpu and node.target in fallback_cpu_random):
                continue
            gm.graph.erase_node(node)
    return gm

replacements = {torch.nn.functional.dropout: lowmem_dropout, torch.rand_like: rand_like}
# Keep track of any replacement functions that use triton random,
# so they can be avoided when fallback_random is set
replacements_using_triton_random = {lowmem_dropout, rand_like}