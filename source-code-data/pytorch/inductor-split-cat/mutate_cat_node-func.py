def mutate_cat_node(match: Match, split_sections: List[int], dim: int):
    if not isinstance(split_sections, (list, tuple)):  # Unnormalized split
        return
    graph = match.graph
    split_node = next(node for node in match.nodes if node.target == torch.split)
    split_input, split_size, split_dim = _get_split_args_default(split_node)
    # if the cat and split have different dims, return
    # Find the next users (i.e. users after the getitem)
    next_users = find_next_users(split_node)
    for cat_user in next_users:
        if cat_user.target == torch.cat:
            cat_dim = get_arg_value(cat_user, 1, "dim") or 0
            # check that all getitems in the cat_user from the same node
            # check the input of the cat has all getitem from the split
            if split_dim != cat_dim or not has_same_parent_node(cat_user):
                continue
            # find the index of getitems to be cat
            indices, idx_to_getitem = [], {}
            for getitem in cat_user.args[0]:  # type: ignore[union-attr]
                indices.append(getitem.args[1])  # type: ignore[union-attr]
                idx_to_getitem[getitem.args[1]] = getitem  # type: ignore[union-attr]
            # the gettitems to be merged must be consecutive, otherwise
            # returned sliced tensor could be wrong
            if not is_sorted_and_consecutive(indices):
                continue
            optimization() # Trigger here