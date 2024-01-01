
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_branch = torch.nn.Conv2d(8, 8, 1, stride=1, padding=1)
        self.bn = torch.nn.BatchNorm2d(8)
    def forward(self, x1):
        v1 = self.conv(x1)
        v1 = v1 + 1
        v1 = F.sigmoid(v1)
        v1 = self.bn(v1)
        v1 = torch.add(v1, 1)
        return v1

# Inputs to the model
x1 = torch.randn(1, 8, 1, 1)
