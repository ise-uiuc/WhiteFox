
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(8, 8, 1, stride=1, padding=1)
    def forward(self, x1):
        v1 = torch.nn.functional.pad(x1, (1, 1, 1, 1), value=-3)
        v2 = self.conv(v1)
        v3 = v2 + 3
        v4 = torch.nn.functional.relu6(v3)
        v5 = v4 / 6
        return v5
# Inputs to the model
x1 = torch.randn(1, 8, 64, 64)
