CallFunction(
    aten.cat, ListOf(CallFunction(aten.addmm, Arg(), Arg(), Arg())), Arg()
)