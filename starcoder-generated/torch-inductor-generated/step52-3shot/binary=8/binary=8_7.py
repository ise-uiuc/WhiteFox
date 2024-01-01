
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.ConvTranspose2d(3, 8, 1, stride=1, padding=1)
        self.conv2 = torch.nn.ConvTranspose2d(1, 8, 1, stride=1, padding=1)
        self.bn1 = torch.nn.BatchNorm2d(8)
        self.bn2 = torch.nn.BatchNorm2d(8)
        self.bn3 = torch.nn.BatchNorm2d(8)
        self.swish1 = Swish()
        self.swish2 = torch.nn.SiLU()
    def forward(self, x1, x2):
        v1 = self.conv1(x1)
        v2 = self.conv2(x2)
        v3 = v1 + v2
        v4 = self.bn1(v3)
        v5 = self.bn2(v4)
        v6 = v4 * v5
        v7 = self.swish1(v6)
        v8 = self.swish2(v7)
        return v8
# Inputs to the model

x1 = torch.randn(1, 3, 128, 128)
x2 = torch.randn(1, 1, 128, 128)
