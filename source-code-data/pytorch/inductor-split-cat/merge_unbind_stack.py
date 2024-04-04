getitem_unbind = ListOf(
    GetItem(
        CallFunction(
            torch.unbind,
            KeywordArg("unbind_input"),
            dim=KeywordArg("dim"),
            _users=MULTIPLE,
        ),
        Ignored(),
        _users=MULTIPLE,
    ),
    partial=True,
)

@register_graph_pattern(
    CallFunction([torch.stack, torch.cat], getitem_unbind, Ignored(), _users=MULTIPLE),
    pass_dict=unbind_stack_pass,
    extra_check=config_flag("split_cat_fx_passes"),
)
@register_graph_pattern(
    CallFunction(
        [torch.stack, torch.cat], getitem_unbind, dim=Ignored(), _users=MULTIPLE
    ),
    pass_dict=unbind_stack_pass,
    extra_check=config_flag("split_cat_fx_passes"),
)
@register_graph_pattern(
    CallFunction(
        [torch.stack, torch.cat], tensors=getitem_unbind, dim=Ignored(), _users=MULTIPLE
    ),
    pass_dict=unbind_stack_pass,
    extra_check=config_flag("split_cat_fx_passes"),
)