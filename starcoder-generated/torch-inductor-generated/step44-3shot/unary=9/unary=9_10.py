
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(8, 16, 1, stride=1, padding=1)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = v1 + 3
        v3 = v2.clamp(min=0, max=6)
        v4 = v3 / 6
        v5 = self.conv2(v4)
        return v5
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
