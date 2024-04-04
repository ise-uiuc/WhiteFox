def is_node_meta_valid(node: Optional[torch.fx.Node]):
    if node is None:
        return True
    if "example_value" not in node.meta:
        return False
    return True


def is_linear_node_can_be_fused(node: torch.fx.Node):
    input = get_arg_value(node, 0, "input")
    weight = get_arg_value(node, 1, "weight")
    return (
        is_node_meta_valid(node)
        and is_node_meta_valid(input)
        and is_node_meta_valid(weight)
        and len(input.meta["example_value"].shape) == 2
        and len(weight.meta["example_value"].shape) == 2
    )


class PreGradBatchLinearFusion(BatchFusion):
    """
    Batch linear fusion in pre grad pass.
    Fuse linear with same size with torch.baddmm
    """

    def _getitem_args(self, getitem_node: torch.fx.Node):
        if getitem_node.target != operator.__getitem__ or (
            getitem_node.op != "call_function"
        ):
            return None
        return getitem_node.args[0]

    def match(self, node: torch.fx.Node):
        if CallFunctionVarArgs(torch.nn.functional.linear).match(
            node
        ) and is_linear_node_can_be_fused(node):
            input = get_arg_value(node, 0, "input")
            weight = get_arg_value(node, 1, "weight")
            bias = get_arg_value(node, 2, "bias")
            group_key = (
                "batch_linear_pre_grad",
                self._getitem_args(input),
                str(input.meta["example_value"].shape),
                str(weight.meta["example_value"].shape),
                bias is None,
            )
        else:
            group_key = None
        return group_key