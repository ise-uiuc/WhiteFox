
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(5, 7, 1, stride=1, padding=1, groups=2)
        self.conv2 = torch.nn.Conv2d(5, 7, 1, stride=1, padding=1, groups=2)
    def forward(self, x1, x2):
        v1 = self.conv(x1)
        v2 = self.conv2(x2)
        return v1 + v2
# Inputs to the model
x1 = torch.randn(1, 5, 8, 8)
x2 = torch.randn(1, 5, 8, 8)
