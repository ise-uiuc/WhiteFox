
class Model2(torch.nn.Module):
    def __init__(self):
        super().__init__()
        x: int = 256
        y: int = 256
        z: int = 256
        self.conv = torch.nn.ConvTranspose2d(z, x // 2, 2, stride=2, padding=0)
        self.conv2 = torch.nn.ConvTranspose2d(x // 2, y, 1, stride=1, padding=0)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = torch.sigmoid(v1)
        v3 = v1 * v2
        v4 = self.conv2(v3)
        return (v3, v4)
# Inputs to the model
x1 = torch.randn(1, 256, 38, 38)
