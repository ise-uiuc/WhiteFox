
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = v1.add_(3)
        v3 = torch.nn.functional.relu6(v2)
        v4 = v1 * v3
        v5 = v4.div_(6)
        return v5
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
