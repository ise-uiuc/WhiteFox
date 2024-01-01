_gelu_fusion_2(u) for u in _computation_user_4

def _gelu_fusion_2(computation_call):
    return CallFunction(
        aten.mul,
        CallFunction(aten.mul, computation_call, 0.5),
        CallFunction(
            aten.add,
            CallFunction(
                aten.tanh,
                CallFunction(
                    aten.mul,
                    CallFunction(
                        aten.add,
                        computation_call,
                        CallFunction(
                            aten.mul,
                            CallFunction(
                                aten.mul,
                                CallFunction(
                                    aten.mul, computation_call, computation_call
                                ),
                                computation_call,
                            ),
                            0.044715,
                        ),
                    ),
                    0.7978845608028654,
                ),
            ),
            1,
        ),
    )

_computation_user_4 = [
    CallFunction(mkldnn._convolution_pointwise.default, *_conv_args, _users=4),
    CallFunction(mkldnn._linear_pointwise.default, *_linear_args, _users=4),
    CallFunction(
        mkldnn._convolution_transpose_pointwise.default,
        *_conv_transpose_args,
        _users=4,
    ),
]