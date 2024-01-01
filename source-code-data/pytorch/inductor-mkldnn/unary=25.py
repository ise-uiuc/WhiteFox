CallFunction(
    aten.where,
    CallFunction(
        aten.gt,
        CallFunction(mkldnn._linear_pointwise.default, *_linear_args, _users=3),
        0,
    ),
    CallFunction(mkldnn._linear_pointwise.default, *_linear_args, _users=3),
    CallFunction(
        aten.mul,
        CallFunction(mkldnn._linear_pointwise.default, *_linear_args, _users=3),
        KeywordArg("negative_slope"),
    ),
)
