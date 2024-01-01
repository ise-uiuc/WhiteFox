
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(16, 16, 2)
        self.bn1 = torch.nn.BatchNorm2d(16)
        self.conv2 = torch.nn.Conv2d(16, 16, 2)
        self.bn2 = torch.nn.BatchNorm2d(16)
    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.bn1(x1)
        x4 = self.bn1(x2)
        x3 = self.conv2(x4)
        x5 = self.bn2(x3)
        x6 = self.bn2(x4)
        m = []
        for _ in range(torch.randint(1, 5, (1,)).item()):
            m.append(x5)
        x7 = (m[0] + m[1] + x1 + x3) + x5 * x6 + torch.cat([x1, x3], dim=1)
        return x7
# Inputs to the model
x = torch.randn(1, 16, 4, 4)
