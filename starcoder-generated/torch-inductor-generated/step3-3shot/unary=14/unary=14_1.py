
class Model3(torch.nn.Module):
    def __init__(self):
        super().__init__()
        x: int = 8
        y: int = 8
        z: int = 3
        self.conv = torch.nn.ConvTranspose2d(z, x // 2, 2, stride=2, padding=1)
        self.conv2 = torch.nn.ConvTranspose2d(y, y // 2, 3, stride=1, padding=0)
        self.conv3 = torch.nn.ConvTranspose2d(y // 2, y, 1, stride=1, padding=0)
        self.linear = torch.nn.Linear(2, 1)
    def forward(self, x1, x2, x3):
        v1 = self.conv(x1)
        v2 = torch.sigmoid(v1)
        v3 = v1 * v2
        v4 = self.conv2(v3)
        v5 = self.conv3(v4)
        v6 = torch.sigmoid(v5)
        v7 = v5 * v6
        y = torch.cat((v7, x2, x3), -3)
        y = torch.abs(y)
        y = torch.sigmoid(y)
        y = self.linear(y)
        return y
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
x2 = torch.randn(1, 4, 16, 16)
x3 = torch.randn(1, 10, 32, 32)
