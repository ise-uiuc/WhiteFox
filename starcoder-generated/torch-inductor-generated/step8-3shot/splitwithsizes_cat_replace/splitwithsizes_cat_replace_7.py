
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, tensor):
        t1 = tensor * 1
        splits = torch.split(t1, split_sizes=3, dim=-1)
        t2 = torch.cat([t for t in splits[0:2]], dim=-1)
        t3 = torch.cat([t for t in splits[1:3]], dim=-1)
        t4 = t2 * t3
        return t4 

# Initializing the model
m = Model()

# Inputs to the model
x = torch.randn(1, 6, 64, 64)
