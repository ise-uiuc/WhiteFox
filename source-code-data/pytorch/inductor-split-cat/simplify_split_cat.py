getitem_split = ListOf(
    CallFunction(
        operator.getitem,
        TorchSplit(
            Ignored(),
            KeywordArg("split_sections"),
        ),
        Ignored(),
        _users=MULTIPLE,
    ),
    partial=True,
)

@register_graph_pattern(
    CallFunction(
        [torch.stack, torch.cat],
        tensors=getitem_split,
        dim=Ignored(),
        _users=MULTIPLE,
    ),
    pass_dict=split_cat_pass,
    extra_check=config_flag("split_cat_fx_passes"),
)
@register_graph_pattern(
    CallFunction(
        [torch.stack, torch.cat],
        getitem_split,
        dim=Ignored(),
        _users=MULTIPLE,
    ),
    pass_dict=split_cat_pass,
    extra_check=config_flag("split_cat_fx_passes"),
)
@register_graph_pattern(
    CallFunction(
        [torch.stack, torch.cat],
        getitem_split,
        Ignored(),
        _users=MULTIPLE,
    ),
    pass_dict=split_cat_pass,
    extra_check=config_flag("split_cat_fx_passes"),
)
def simplify_split_cat(match: Match, split_sections: List[int], dim: int):
    if not isinstance(split_sections, (list, tuple)):  # Unnormalized split
        return
    optimization() # Trigger here