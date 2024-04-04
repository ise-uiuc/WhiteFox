class BatchPointwiseOpsPreGradFusion(BatchPointwiseOpsFusionFactory):
    """
    Batch poinwise ops (e.g., sigmoid, relu, tanh) fusion in pre grad pass.
    We fuse it in random place, and the introduced stack node may be merged in split cat.
    """

    def __init__(self, op, **kwargs):
        super().__init__(op, **kwargs)
        self.op = op

    def match(self, node: torch.fx.Node):
        input = get_arg_value(node, 0, "input")
        if CallFunctionVarArgs(self.op).match(node) and is_node_meta_valid(node):
            # for relu op, we also use the inplace to construct the key
            group_key = (
                "batch_" + self.op.__name__.lower() + "_pre_grad",
                str(input.meta["example_value"].shape),
                str(node.kwargs.get("inplace", False)),
            )
        else:
            group_key = None
        return group_key