def randperm_index_pattern(x, slice_shape):
    index = torch.randperm(x.shape[0], device=x.device)[:slice_shape]
    return torch.ops.aten.index(x, (index,)), index