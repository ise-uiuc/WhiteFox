
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x1):
        t1 = torch.cat([x1, x1], dim=1)
        t2 = t1[:, -1:]
        t3 = t2[:, :250]
        return torch.cat([t1, t3], dim=1)

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 250, 64, 64)
