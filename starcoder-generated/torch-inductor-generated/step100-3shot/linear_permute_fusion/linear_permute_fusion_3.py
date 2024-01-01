
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(1, 2, kernel_size=(3, 3))
        self.bn = torch.nn.BatchNorm2d(1)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = v1.view(1, 3, 7)
        v3 = self.bn(v2)
        v4 = self.bn(v2)
        v5 = v4.permute(0, 2, 1)
        return v5
# Inputs to the model
x1 = torch.randn(1, 1, 4, 4)
