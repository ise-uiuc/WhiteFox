CallFunction(
    aten.div,
    CallFunction(
        aten.mul,
        CallFunction(mkldnn._convolution_pointwise.default, *_conv_args, _users=2),
        CallFunction(
            aten.clamp_max,
            CallFunction(
                aten.clamp_min,
                CallFunction(
                    aten.add,
                    CallFunction(
                        mkldnn._convolution_pointwise.default, *_conv_args, _users=2
                    ),
                    3,
                ),
                0,
            ),
            6,
        ),
    ),
    6,
)
