
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        x: int = 8
        y: int = 8
        z: int = 3
        self.conv = torch.nn.ConvTranspose2d(z, y, 1, stride=1, padding=0)
        self.conv_transpose = torch.nn.ConvTranspose2d(y, x * 8, 1, stride=1, padding=0)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = torch.sigmoid(v1)
        v3 = v1 * v2
        v4 = self.conv_transpose(v3)
        return (v3, v4)
# Inputs to the model
x1 = torch.randn(1, 3, 128, 128)
