def _op_not_broadcasting_with_conv(weight_tensor, other_tensor):
    # According to opDoesNotBroadCastWithConv of frozen_conv_folding.cpp
    weight_shape = weight_tensor.shape
    other_shape = other_tensor.shape
    if len(weight_shape) < len(other_shape):
        return False
    if len(weight_shape) == len(other_shape) + 1:
        # weight shape is [o, i, *], other_shape is [o, 1...].
        for i in reversed(range(len(other_shape))):
            if i == 0 and weight_shape[0] == other_shape[i]:
                continue
            if other_shape[i] != 1:
                return False
    else:
        # weight shape is [o, i, *], other_shape is [1, i, *]
        for i in reversed(range(len(other_shape))):
            if i == 1 and weight_shape[0] == other_shape[i]:
                continue
            if other_shape[i] != 1:
                return False
    return True

def _check_conv_and_broadcast_op(conv_node, other):
    # According to checkConvAndBroadcastingOpPreConditions of frozen_conv_folding.cpp.
    # conv.weight
    if conv_node.args[1].op != "get_attr":
        return False
    # conv.bias
    if conv_node.args[1] is not None and conv_node.args[1].op != "get_attr":
        return False
    if (
        not isinstance(other, int)
        and not isinstance(other, float)
        and other.op != "get_attr"
    ):
        return False

    if not len(conv_node.args[1].users) == 1:
        return False

    weight_meta_value = conv_node.args[1].meta.get("val")
    if weight_meta_value is None:
        return False
    # Avoid fusing op that causes type promotion
    # restricting to float avoids int/float difficulties with scalar overload
    if not weight_meta_value.is_floating_point():
        return False
    if isinstance(other, torch.fx.Node) and other.op == "get_attr":
        other_meta_value = other.meta.get("val")
        if not other_meta_value.is_floating_point():
            return False
        if (
            torch.promote_types(other_meta_value.dtype, weight_meta_value.dtype)
            != weight_meta_value.dtype
        ):
            if not conv_node.meta.get("_allow_conv_mixed_dtype_folding", False):
                return False

            if (
                other_meta_value.dtype != torch.float
                and weight_meta_value.dtype not in (torch.float16, torch.bfloat16)
            ):
                return False

        if not _op_not_broadcasting_with_conv(weight_meta_value, other_meta_value):
            return False
    else:
        # TODO: support scalar case
        return False

    return True

def _is_foldable_pattern(match):
    binary_node = match.output_node()
    computation_node = binary_node.args[0]
    other = binary_node.args[1]
    if binary_node.args[0].target not in _computation_ops:
        computation_node = binary_node.args[1]
        other = binary_node.args[0]
    if binary_node.args[0].target == aten.convolution.default:
        return _check_conv_and_broadcast_op(computation_node, other)

    return False