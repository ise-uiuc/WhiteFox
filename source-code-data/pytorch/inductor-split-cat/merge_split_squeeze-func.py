def merge_split_squeeze(
    match: Match, split_input: torch.fx.Node, split_sizes: List[int], dim: int
):
    graph = match.graph
    split = next(node for node in match.nodes if node.target == torch.split)
    if not all(s == 1 for s in split_sizes):
        return
    if isinstance(dim, Sequence):
        return
    next_users = find_next_users(split)
    if not all(node.target == torch.squeeze for node in next_users):
        return
    optimization() # Trigger here