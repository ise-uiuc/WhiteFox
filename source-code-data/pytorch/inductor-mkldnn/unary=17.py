CallFunction(
    aten.relu,
    CallFunction(
        mkldnn._convolution_transpose_pointwise.default,
        *_conv_transpose_args,
        _users=1,
    ),
)
