
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x1):
        v1 = x1.reshape(x1.shape[0], -1)
        v2 = torch.nn.functional.linear(v1, torch.eye(512, dtype=torch.float), bias=torch.zeros(512, dtype=torch.float))
        v3 = v2.flatten(start_dim=1)
        v4 = v3 + 3
        v5 = torch.clamp(v4, min=0)
        v6 = torch.clamp(v5, max=6)
        v7 = v6 / 6
        return v7

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 100, 512)
v1 = x1.reshape(x1.shape[0], -1)
