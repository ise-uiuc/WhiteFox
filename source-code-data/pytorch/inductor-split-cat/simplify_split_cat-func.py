def simplify_split_cat(match: Match, split_sections: List[int], dim: int):
    if not isinstance(split_sections, (list, tuple)):  # Unnormalized split
        return
    optimization() # Trigger here