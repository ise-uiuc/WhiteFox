CallFunction(
    aten.mul,
    CallFunction(
        mkldnn._convolution_transpose_pointwise.default,
        *_conv_transpose_args,
        _users=2,
    ),
    CallFunction(
        aten.sigmoid,
        CallFunction(
            mkldnn._convolution_transpose_pointwise.default,
            *_conv_transpose_args,
            _users=2,
        ),
    ),
)
