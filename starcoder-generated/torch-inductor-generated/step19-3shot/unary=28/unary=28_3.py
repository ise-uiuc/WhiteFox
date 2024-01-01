
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        v1 = torch.linear(x)
        v2 = torch.clamp_min(v1, min_value=3)
        return torch.clamp_max(v2, max_value=-3)

# Intializing the model
m = Model()

# Input to the model
x = torch.randn(1, 3, 32, 32)
