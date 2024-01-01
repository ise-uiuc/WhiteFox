
def is_valid_splitwithsizes_cat(graph):
    nodes = graph.nodes()

    splitwithsizes_kinds = ["aten::split_with_sizes", "split_with_sizes"]
    cat_kinds = ["aten::cat", "cat"]
    getitem_kinds = ["aten::operator.getitem", "operator.getitem"]

    splitwithsizes_nodes = [n for n in nodes if n.kind() in splitwithsizes_kinds]
    cat_nodes = [n for n in nodes if n.kind() in cat_kinds]
    getitem_nodes = [n for n in nodes if n.kind() in getitem_kinds]

    if len(splitwithsizes_nodes)!= 1:
        return False
    if len(cat_nodes)!= 1:
        return False
    s = splitwithsizes_nodes[0]
    c = cat_nodes[0]
    for i in range(s.inputs().__len__()):
        if (s.inputs()[i].unique()!= c.inputs()[i].unique()):
            return False
    if not s.outputs().__len__() == (c.inputs().__len__() - 1):
        return False
    
    for g in getitem_nodes:
        firsts = [a.unique() for a in s.outputs()]
        seconds = [g.inputs()[0].unique()]
        if set(firsts)!= set(seconds):
            return False
        
    return True

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 4, 3, stride=1, padding=1)
 
    def forward(self, x):
        v1 = x.permute(0, 1, 3, 2)
        v2 = v1.split(1, 3)
        v3 = v2[0] + v2[1]
        v4 = v3.permute(2, 3, 0, 1)
        return v4

# Initializing the model
m = Model()

# Inputs to the model
x = torch.randn(5, 3, 8, 8)

# If the requirements are not met, return the original model
# is_valid = is_valid_cat_permute(x)
# if is_valid:
