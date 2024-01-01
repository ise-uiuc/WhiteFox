
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(8, 8, 3, stride=2, padding=1)
        self.conv2 = torch.nn.Conv2d(8, 8, 3, stride=1, padding=1)
    def forward(self, x1, x2):
        v1 = self.conv1(x1)
        v2 = self.conv2(x2)
        v3 = F.interpolate(v1, scale_factor=2.0, mode='nearest')
        v4 = v3 + v2
        return v4
# Inputs to the model
x1 = torch.randn(1, 16, 32, 32)
x2 = torch.randn(1, 8, 64, 64)
