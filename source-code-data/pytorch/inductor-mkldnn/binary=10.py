CallFunction(
    aten.add,
    KeywordArg("other"),
    CallFunction(mkldnn._linear_pointwise.default, *_linear_args, _users=1),
)
