class PostGradBatchLinearFusion(BatchFusion):
    """
    Fuse ops in a batch way in post grad (aten level).
    """

    def _addmm_node_can_be_fused(self, node: torch.fx.Node) -> bool:
        return (
            node.kwargs.get("beta", 1.0) == 1.0 and node.kwargs.get("alpha", 1.0) == 1.0  # type: ignore[return-value]
        )

    def _is_input_2d(self, input: torch.fx.Node) -> bool:
        input_shapes = input.meta["tensor_meta"].shape
        return (
            len(input_shapes) == 2
            and isinstance(input_shapes[0], int)
            and isinstance(input_shapes[1], int)
        )

    def match(self, node: torch.fx.Node) -> Optional[Tuple[str, int, int, int, bool]]:
        if CallFunctionVarArgs(aten.mm).match(node):
            input_m, weight_m = node.args
            bias_m = None

        elif CallFunctionVarArgs(aten.addmm.default).match(
            node
        ) and self._addmm_node_can_be_fused(node):
            bias_m, input_m, weight_m = node.args
        else:
            return None

        # only handle the cases where inputs are 2D tensors
        if not self._is_input_2d(input_m) or not self._is_input_2d(weight_m):  # type: ignore[arg-type]
            return None
        m, k = input_m.meta["tensor_meta"].shape  # type: ignore[union-attr]
        n = weight_m.meta["tensor_meta"].shape[1]  # type: ignore[union-attr]
        batch_key = ("batch_linear", m, k, n, bias_m is not None)
        return batch_key
