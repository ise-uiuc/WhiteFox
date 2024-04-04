# These patterns do 2 things
# 1. Since we know that index is completely unique, we can codegen it using
# stores instead of atomic adds, which is quite a bit faster.
# 2. Also, since we are guaranteed that they are completely within bounds,
# we can use unsafe indexing and skip debug asserts
def randperm_index_add_pattern(x, y):
    index = torch.randperm(x.shape[0], device=x.device)[: y.shape[0]]
    return torch.index_add(x, dim=0, source=y, index=index), index