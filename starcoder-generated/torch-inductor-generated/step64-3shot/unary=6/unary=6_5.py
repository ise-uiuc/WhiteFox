
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.convbnrelu = torch.nn.Sequential(
                          torch.nn.Conv2d(3, 6, 1, stride=1, padding=1),
                          torch.nn.BatchNorm2d(6),
                          torch.nn.ReLU6()
        )
        self.conv = torch.nn.Conv2d(6, 3, 1, stride=1, padding=1)
    def forward(self, x1):
        v1 = self.convbnrelu(x1)
        v2 = self.convbnrelu(v1)
        v3 = self.convbnrelu(v2)
        v4 = self.conv(v3)
        return torch.mean(v4, dim=1)
# Inputs to the model
x1 = torch.randn(1, 3, 256, 256)
