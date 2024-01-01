CallFunction(
    aten.add,
    KeywordArg("other"),
    CallFunction(mkldnn._convolution_pointwise.default, *_conv_args, _users=1),
)
