
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(416, 1)

    def forward(self, x1):
        v0 = x1.permute(0, 2, 3, 1)
        v1 = self.linear(v0)
        v2 = torch.clamp_min(v1, min=0.0)
        return torch.clamp_max(v2, max=0.1)

# Initializing the model
m = Model()

# Inputs to the model
__input__ = torch.randn(16, 3, 336, 336)
x1 = __input__.to(device)
