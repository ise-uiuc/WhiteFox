
class Model2(torch.nn.Module):
    def __init__(self):
        super().__init__()
        x: int = 8
        y: int = 8
        z: int = 3
        self.conv = torch.nn.ConvTranspose2d(x, y, 1, stride=1, padding=0)
        self.conv2 = torch.nn.ConvTranspose2d(y + x, y, 1, stride=1, padding=0)
        self.conv3 = torch.nn.ConvTranspose2d(z, x, 1, stride=1, padding=0)
    def forward(self, x1, x2):
        v1 = self.conv(x1)
        v2 = torch.sigmoid(v1)
        v3 = v1 * v2
        v4 = torch.cat((x1, x2), dim=1)
        v5 = self.conv2(v4)
        v6 = torch.sigmoid(v5)
        v7 = v4 + v6
        v8 = self.conv3(v7)
        return (v5, v6, v8)
# Inputs to the model
x1 = torch.randn(1, 8, 64, 64)
x2 = torch.randn(1, 3, 64, 64)
