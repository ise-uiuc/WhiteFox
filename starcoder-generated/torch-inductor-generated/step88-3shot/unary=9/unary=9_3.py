
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.relu6 = torch.nn.ReLU6()
        self.conv = nn.Conv2d(3, 16, 3, stride=1, padding=1)
    def forward(self, x1):
        x1 = self.conv(x1)
        x2 = x1 + 3
        x3 = torch.clamp(x2, 0, 6)
        x4 = x3 / 6
        return x4
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
