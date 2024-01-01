
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        split_sizes = [17, 16, 4]
        x1, x2, x3 = torch.split(x, split_sizes, dim=0)
        x4 = torch.cat([x2, x3], dim=0)
        x5 = torch.cat([x1, x4], dim=0)
        ret = torch.cat([x5, x4, x3], dim=0)
        return tuple(ret.shape)

# Initializing the model
m = Model()

# Inputs to the model
x = torch.randn(33, 3)
