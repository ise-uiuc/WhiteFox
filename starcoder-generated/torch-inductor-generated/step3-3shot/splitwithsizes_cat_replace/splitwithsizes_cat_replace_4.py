
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x1):
        v1 = torch.split(x1, 1)
        v2 = torch.cat([v1[i] for i in range(1)], 0)
        return v2

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(4, 1)
