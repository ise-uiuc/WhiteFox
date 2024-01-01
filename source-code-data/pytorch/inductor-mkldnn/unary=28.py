CallFunction(
    aten.clamp_max,
    CallFunction(
        aten.clamp_min,
        CallFunction(mkldnn._linear_pointwise.default, *_linear_args, _users=1),
        KeywordArg("min_value"),
    ),
    KeywordArg("max_value"),
)
