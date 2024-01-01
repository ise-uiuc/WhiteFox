
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x1):
        split = torch.split(x1, split_sizes=[12, 8, 95, 111], dim=3)
        v2 = torch.cat([split[i] for i in range(len(split_sizes))], dim=3)
        return v2

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 1, 56, 149)
