CallFunction(
    aten.div,
    CallFunction(
        aten.clamp_max,
        CallFunction(
            aten.clamp_min,
            CallFunction(
                aten.add,
                CallFunction(mkldnn._linear_pointwise.default, *_linear_args, _users=1),
                3,
            ),
            0,
        ),
        6,
    ),
    6,
)
