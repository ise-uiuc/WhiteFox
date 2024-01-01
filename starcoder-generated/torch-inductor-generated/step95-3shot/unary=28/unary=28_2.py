
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, v1):
        v2 = torch.nn.functional.linear(v1, 10)
        v3 = torch.clamp_min(v2, 1.23)
        v4 = torch.clamp_max(v3, 3.45)
        return v4

# Initializing the model
m = Model()

# Inputs to the model
v1 = torch.randn(4, 10)
