def merge_stack_tahn_unbind(match: Match, split_sections: List[int], dim: int):
    if not isinstance(split_sections, (list, tuple)):  # Unnormalized split
        return
    graph = match.graph
    split_node = next(node for node in match.nodes if node.target == torch.split)
    split_input, split_size, split_dim = _get_split_args_default(split_node)
    # Find the next users (i.e. users after the getitem)
    next_users = find_next_users(split_node)
    # 'immutable_list' object does not support mutation. Create a new copy of it
    split_sections = list(split_sections)
    for user in next_users:
        # stack user only has one user
        if user.target == torch.stack:
            stack_dim = get_arg_value(user, 1, "dim") or 0
            unbind_user = find_next_users(user)[0]
            if unbind_user.target != torch.unbind:
                continue
            unbind_dim = get_arg_value(unbind_user, 1, "dim") or 0
            # stack and unbind should have the same dim
            # check the all getitems in the user from the same node
            # check all the getitems only has single user
            if (
                stack_dim != unbind_dim
                or not has_same_parent_node(user)
                or not all(len(arg.users) == 1 for arg in user.args[0])  # type: ignore[union-attr]
            ):
                continue
            # find the index of getitems to be stacked
            indices = []
            split_sections_for_unbind = []
            for arg in user.args[0]:  # type: ignore[union-attr]
                indices.append(arg.args[1])  # type: ignore[union-attr]
                split_sections_for_unbind.append(split_sections[arg.args[1]])  # type: ignore[union-attr]
            # the gettitems to be merged must be consecutive, otherwise
            # returned sliced tensor could be wrong
            if not is_sorted_and_consecutive(indices):
                continue
            optimization() # Trigger here