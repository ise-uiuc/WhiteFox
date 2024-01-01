
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1):
        v1 = torch.clamp(x1, min=0)
        v2 = v1 / 4
        v3 = torch.clamp(v2, min=0)
        v4 = v3 / 2
        v5 = torch.clamp(v4, min=0)
        v6 = v5 / 6
        return v6
# Inputs to the model
x1 = torch.randn(1, 56, 48, 48)
