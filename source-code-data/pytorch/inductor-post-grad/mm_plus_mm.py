CallFunction(
    aten.add,
    CallFunction(aten.mm, Arg(), Arg()),
    CallFunction(aten.mm, Arg(), Arg()),
)