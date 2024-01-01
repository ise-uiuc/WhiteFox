
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1):
        v1 = x1 + 3
        v2 = torch.clamp(v1, min=0)
        v3 = torch.clamp(v2, max=6)
        v4 = v1 - 3
        v5 = torch.clamp(v4, min=0)
        v6 = torch.clamp(v5, max=6)
        return 2 * v3 / v6
# Inputs to the model
x1 = torch.randn(1, 3, 128, 128)
