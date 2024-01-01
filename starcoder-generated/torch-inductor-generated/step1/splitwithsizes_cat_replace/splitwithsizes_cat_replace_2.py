
class Model(torch.nn.Module):
    def __init__(self, dim=1):
        super().__init__()
        self.dim = dim
 
    def forward(self, x):
        split_sizes = [3, 2, 5]
        parts = torch.split(x, split_sizes, self.dim)
        # In this example, `parts` is a list of 3 tensors.
        # The order is from smaller to larger sizes.
        dim = self.dim
        xcat = []
        for i in range(len(parts)):
            xcat.append(parts[i-1-2*i])
        xcat = torch.cat(xcat, dim=dim)
        # The returned value `xcat` would be either
        # the concatenated tensor with `dim=0` or `dim=-1`.
        return xcat
 
def is_valid_splitwithsizes_cat(graph: GraphModule, match: Matcher) -> bool:
    split_nodes = [match.nodes_by_name['split_node']]
    cat_nodes = [match.nodes_by_name['cat_node']]
    getitem_nodes = [
        match.nodes_by_name[f'get{i}_node']
        for i in range(len(split_nodes[0].op.split_sizes))
    ]
    # Check if there is a split with sizes and a catenation node
    if (len(split_nodes)!= 1 or len(cat_nodes)!= 1):
        return False
    # Check if the split and the cat axis are the same
    dim = split_nodes[0].op.dim
    if (dim!= cat_nodes[0].op.dim):
        return False
    # Check if the split parts are used exactly once in the catenation
    used_parts = []
    for node in getitem_nodes:
        if (node.op.index in used_parts):
            return False
        used_parts.append(node.op.index)
    # Check if the part items are arranged in the same order as the split sizes
    split_sizes = split_nodes[0].op.split_sizes
    if (split_sizes!= used_parts):
        return False
    return True


# Initializing the model
m = Model()
gm = symbolic_trace(m)

# Inputs to the model
x = torch.randn(20, 5, 64, 64)
