
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1):
        v1 = (x1 + 3).transpose(0, 1).view(32, 64, 16)
        v2 = v1 - 3
        v3 = torch.clamp_min(v2, 0)
        v4 = torch.clamp_max(v3, 6)
        v5 = v4 / 6
        return x1
# Inputs to the model
x1 = torch.randn(1, 64, 24)
