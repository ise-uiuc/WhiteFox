
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1):
        v1 = x1 + 3
        v2 = torch.clamp_min(v1, 0)
        v3 = torch.clamp_max(v2, 6)
        return v3
# Inputs to the model
x1 = torch.randn(1, 16, 64, 64)
