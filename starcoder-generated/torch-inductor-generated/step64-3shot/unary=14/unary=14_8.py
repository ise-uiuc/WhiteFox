
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1):
        v1 = torch.nn.functional.interpolate(x1, (8, 8))
        v2 = torch.nn.functional.interpolate(v1, (16, 16))
        v3 = torch.clamp(v2, -1, 1)
        return v3
# Inputs to the model
x1 = torch.randn(1, 1, 16, 16)
