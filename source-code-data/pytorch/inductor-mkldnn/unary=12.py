CallFunction(
    aten.mul,
    CallFunction(mkldnn._convolution_pointwise.default, *_conv_args, _users=2),
    CallFunction(
        aten.sigmoid,
        CallFunction(mkldnn._convolution_pointwise.default, *_conv_args, _users=2),
    ),
)
