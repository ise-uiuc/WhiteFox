CallFunction(
    aten.relu,
    CallFunction(
        ops.sub,
        CallFunction(mkldnn._convolution_pointwise.default, *_conv_args, _users=1),
        KeywordArg("other"),
    ),
)
