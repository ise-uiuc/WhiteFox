
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(16, 16, 7, stride=1, padding=3)
        self.conv2 = torch.nn.Conv2d(32, 32, 7, stride=1, padding=3)
    def forward(self, x, x2):
        v1 = self.conv1(x)
        v2 = self.conv2(v1)
        return v2 + x2
# Inputs to the model
x = torch.randn(1, 16, 64, 64)
x2 = torch.randn(3, 32, 56, 56)
