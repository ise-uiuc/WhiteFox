
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(16, 16, 1, 1, 0, groups=2)
        self.conv2 = torch.nn.Conv2d(16, 16, 3, 2, 1, groups=2)
    def forward(self, x1, x2, x3, x4):
        v1 = self.conv1(x1)
        v2 = v1 + x1
        v3 = self.conv2(v1)
        v4 = v3 + x1
        return v4
# Inputs to the model
x1 = torch.randn(1, 16, 89, 89)
x2 = torch.randn(1, 16, 89, 89)
x3 = torch.randn(1, 16, 89, 89)
x4 = torch.randn(1, 16, 89, 89)
