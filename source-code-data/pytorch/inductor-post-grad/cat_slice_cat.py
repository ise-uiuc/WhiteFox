_cat_1 = CallFunction(aten.cat, Arg(), 1, _users=2)
CallFunction(
    aten.cat,
    [
        _cat_1,
        CallFunction(
            aten.slice,
            CallFunction(aten.slice, _cat_1, 0, 0, 9223372036854775807),
            1,
            0,
            KeywordArg("size"),
        ),
    ],
    1,
)