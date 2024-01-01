
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(16, 17, 1, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(17, 18, 1, stride=1, padding=1)
    def forward(self, x1, x2, other):
        v1 = self.conv1(x1)
        v2 = self.conv2(v1)
        v3 = v2 + other
        return v3
# Inputs to the model
x1 = torch.randn(1, 16, 64, 64)
x2 = torch.randn(1, 17, 64, 64)
other = torch.randn(1, 18, 64, 64)
