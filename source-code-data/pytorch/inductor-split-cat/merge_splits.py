TorchSplit(
    CallFunction(
        operator.getitem,
        TorchSplit(
            KeywordArg("first_split_input"),
            KeywordArg("first_split_sections"),
        ),
        Ignored(),
    ),
    KeywordArg("next_split_sections"),
)

class TorchSplit(CallFunction):
    """
    Matches a call to torch.split if it is in a normalized form. Ensures that all users of
    splits are unique getitems.
    """

    def __init__(self, arg, sizes, func=torch.split):
        # using KeywordArg("dim") for `dim` checks they all match
        super().__init__(func, arg, sizes, _users=MULTIPLE, dim=KeywordArg("dim"))

    def _match(self, node: torch.fx.Node, ctx: MatchContext):
        m = super()._match(node, ctx)
        if not m:
            return m
        split_sections = node.args[1]
        if not isinstance(split_sections, (list, tuple)):
            return FailedMatch("split not normalized")
        # check users are all unique getitems
        seen_idxs = set()
        for user in node.users:
            if not CallFunction(operator.getitem, Arg(), Arg()).match(user):
                # This should ideally never happen. Split user should always be a getitem
                return FailedMatch(f"user of split not a getitem: {user}")
            if not isinstance(user.args[1], int):
                return FailedMatch("only integer getitems are handled")
            if user.args[1] in seen_idxs:
                return FailedMatch(f"duplicate getitem {user.args[1]}")
            if user.args[-1] < 0:  # type: ignore[operator]
                # This shouldn't ideally happen as dynamo normalizes indexes to positive
                return FailedMatch("negative index")
            seen_idxs.add(user.args[1])
        optimization() # Trigger here
        return m