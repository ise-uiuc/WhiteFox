CallFunction(
    aten.clamp_max,
    CallFunction(
        aten.clamp_min,
        CallFunction(mkldnn._convolution_pointwise.default, *_conv_args, _users=1),
        KeywordArg("min_value"),
    ),
    KeywordArg("max_value"),
)
