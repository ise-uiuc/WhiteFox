class GroupLinearFusion(GroupFusion):
    def _addmm_node_can_be_fused(self, node: torch.fx.Node):
        input_shape = node.args[1].meta["tensor_meta"].shape  # type: ignore[union-attr]
        weight_shape = node.args[2].meta["tensor_meta"].shape  # type: ignore[union-attr]
        return (
            node.kwargs.get("beta", 1.0) == 1.0
            and node.kwargs.get("alpha", 1.0) == 1.0
            and len(input_shape) == 2
            and len(weight_shape) == 2
            and all(x % 2 == 0 for x in input_shape + weight_shape)
            and all(
                shape <= self.graph_search_options["max_fuse_tensor_size_group_linear"]
                for shape in input_shape + weight_shape
            )
        )

    def _mm_node_can_be_fused(self, node: torch.fx.Node):
        input_shape = node.args[0].meta["tensor_meta"].shape  # type: ignore[union-attr]
        weight_shape = node.args[1].meta["tensor_meta"].shape  # type: ignore[union-attr]
        return (
            len(input_shape) == 2
            and len(weight_shape) == 2
            and all(x % 2 == 0 for x in input_shape + weight_shape)
            and all(
                shape <= self.graph_search_options["max_fuse_tensor_size_group_linear"]
                for shape in input_shape + weight_shape
            )
        )

    def match(self, node: torch.fx.Node) -> Optional[Tuple[str, bool]]:
        if CallFunctionVarArgs(aten.mm.default).match(
            node
        ) and self._mm_node_can_be_fused(node):
            group_key = ("group_linear", True)
        elif CallFunctionVarArgs(aten.addmm.default).match(
            node
        ) and self._addmm_node_can_be_fused(node):
            bias = node.args[0]
            group_key = ("group_linear", bias is None)
        else:
            group_key = None
        return group_key