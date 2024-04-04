class BatchPointwiseOpsPostGradFusion(BatchPointwiseOpsFusionFactory):
    """
    Batch pointwise operator (e.g., add, mul) in post grad pass.
    """

    def __init__(self, op, **kwargs):
        super().__init__(op, **kwargs)
        self.op = op

    def _pointwise_node_can_be_fused(self, node: torch.fx.Node):
        # note: we only consider the case where the inputs are tensors
        # for mixed precision training, we need to make sure the inputs
        # of the aten.cat when do the stack should be the same dtype
        # otherwise, the output of the aten.cat may be not the same as
        # its inputs, and cause dtype not same error in mm or addmm
        input, other = node.args
        return (
            input.meta["tensor_meta"].shape == other.meta["tensor_meta"].shape  # type: ignore[union-attr]
            if hasattr(input, "meta")
            and hasattr(other, "meta")
            and "tensor_meta" in input.meta  # type: ignore[union-attr]
            and "tensor_meta" in other.meta  # type: ignore[union-attr]
            else False
        )

    def match(self, node: torch.fx.Node):
        if CallFunctionVarArgs(self.op).match(
            node
        ) and self._pointwise_node_can_be_fused(node):
            alpha = node.kwargs.get("alpha", 1.0)
            rounding_mode = node.kwargs.get("rounding_mode", None)
            input, other = node.args
            shape = list(input.meta["tensor_meta"].shape)  # type: ignore[union-attr]
            group_key = (
                "batch_" + self.op.__name__.lower() + "_post_grad",
                str(shape),
                str(input.meta["tensor_meta"].dtype),  # type: ignore[union-attr]
                str(other.meta["tensor_meta"].dtype),  # type: ignore[union-attr]
                str(alpha),
                str(rounding_mode),
            )
        else:
            group_key = None
        return group_key