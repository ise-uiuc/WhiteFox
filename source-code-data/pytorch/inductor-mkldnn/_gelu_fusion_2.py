CallFunction(
    aten.mul,
    CallFunction(aten.mul, CallFunction(mkldnn._convolution_pointwise.default, *_conv_args, _users=4), 0.5),
    CallFunction(
        aten.add,
        CallFunction(
            aten.tanh,
            CallFunction(
                aten.mul,
                CallFunction(
                    aten.add,
                    CallFunction(mkldnn._convolution_pointwise.default, *_conv_args, _users=4),
                    CallFunction(
                        aten.mul,
                        CallFunction(
                            aten.mul,
                            CallFunction(
                                aten.mul, CallFunction(mkldnn._convolution_pointwise.default, *_conv_args, _users=4), CallFunction(mkldnn._convolution_pointwise.default, *_conv_args, _users=4)
                            ),
                            CallFunction(mkldnn._convolution_pointwise.default, *_conv_args, _users=4),
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