CallFunction(
    aten.relu,
    CallFunction(
        aten.add,
        CallFunction(mkldnn._linear_pointwise.default, *_linear_args, _users=1),
        KeywordArg("other"),
    ),
)
