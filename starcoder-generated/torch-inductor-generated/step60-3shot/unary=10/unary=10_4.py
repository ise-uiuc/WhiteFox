
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x1):
        v1 = x1.flatten(start_dim=1)
        v2 = torch.relu(v1)
        v3 = v2 + 3
        v4 = torch.clamp(v3, 0, 6)
        v5 = v4 / 6
        return v5

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
