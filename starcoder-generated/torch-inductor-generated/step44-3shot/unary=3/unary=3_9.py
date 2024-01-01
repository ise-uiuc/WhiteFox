
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1):
        v1 = torch.nn.functional.conv1d(x1, torch.ones(7, 64, 3), stride=1, padding=0, dilation=1, groups=1)
        v2 = v1 * 0.5
        v3 = v1 * 0.7071067811865476
        v4 = torch.nn.functional.erf(v3)
        v5 = v4 + 1
        v6 = v2 * v5
        return v6
# Inputs to the model
x1 = torch.randn(1, 64, 167)
