
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1):
        v1 = x1 + 3
        v2 = torch.clamp(v1, 0, 6)
        v3 = v2 / 6
        return v3
# Inputs to the model
x1 = torch.randn(1, 3, 256, 256)
