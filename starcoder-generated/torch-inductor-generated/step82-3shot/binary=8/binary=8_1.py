
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
    def forward(self, x1, x2, other):
        v1 = self.conv1(x1)
        v2 = self.conv2(x2)
        v3 = v1 + v2 + other
        return v3
# Inputs to the model
x1 = torch.randn(3, 3, 32, 32)
x2 = torch.randn(3, 3, 32, 32)
other = torch.randn(3, 3, 32, 32)
