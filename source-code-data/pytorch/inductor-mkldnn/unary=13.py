CallFunction(
    aten.mul,
    CallFunction(mkldnn._linear_pointwise.default, *_linear_args, _users=2),
    CallFunction(
        aten.sigmoid,
        CallFunction(mkldnn._linear_pointwise.default, *_linear_args, _users=2),
    ),
)
