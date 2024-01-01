CallFunction(
    aten.where,
    CallFunction(
        aten.gt,
        CallFunction(mkldnn._convolution_pointwise.default, *_conv_args, _users=3),
        0,
    ),
    CallFunction(mkldnn._convolution_pointwise.default, *_conv_args, _users=3),
    CallFunction(
        aten.mul,
        CallFunction(mkldnn._convolution_pointwise.default, *_conv_args, _users=3),
        KeywordArg("negative_slope"),
    ),
)
