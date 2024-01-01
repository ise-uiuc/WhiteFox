
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv2 = torch.nn.Conv2d(3, 3, 3)
        self.conv1_0 = torch.nn.Conv2d(3, 3, 3)
        self.conv1_1 = torch.nn.Conv3d(3, 3, 3)
        self.bn = torch.nn.BatchNorm2d(3)
    def forward(self, x1):
        s = self.conv1_0(x1)
        t = self.conv2(self.conv1_0(x1))
        y = self.conv1_1(s)
        y = self.bn(y)
        y = self.bn(y)
        return torch.abs(torch.sum(torch.norm(torch.cat((t, y), 1))))
# Inputs to the model
x = torch.randn(1, 3, 6, 6)
