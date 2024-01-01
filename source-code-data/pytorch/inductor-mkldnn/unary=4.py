CallFunction(
    aten.mul,
    CallFunction(
        aten.mul,
        CallFunction(mkldnn._linear_pointwise.default, *_linear_args, _users=2),
        0.5,
    ),
    CallFunction(
        aten.add,
        CallFunction(
            aten.erf,
            CallFunction(
                aten.mul,
                CallFunction(mkldnn._linear_pointwise.default, *_linear_args, _users=2),
                0.7071067811865476,
            ),
        ),
        1,
    ),
)
