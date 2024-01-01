CallFunction(
    aten.mul,
    CallFunction(
        aten.mul,
        CallFunction(mkldnn._convolution_pointwise.default, *_conv_args, _users=2),
        0.5,
    ),
    CallFunction(
        aten.add,
        CallFunction(
            aten.erf,
            CallFunction(
                aten.mul,
                CallFunction(
                    mkldnn._convolution_pointwise.default, *_conv_args, _users=2
                ),
                0.7071067811865476,
            ),
        ),
        1,
    ),
)
