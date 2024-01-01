CallFunction(
    aten.add,
    CallFunction(aten.mm, Arg(), Arg()),
    KeywordArg("inp"),
)
