
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(512, 1, 3, stride=1, padding=1, groups=2)
        self.conv_transpose16 = torch.nn.ConvTranspose2d(32, 128, 3, stride=1, padding=1)
    def forward(self, x1, x2):
        v1 = self.conv(x1)
        v2 = self.conv_transpose16(x2)
        v3 = torch.sigmoid(v1 + v2)
        return v3
# Inputs to the model
x1 = torch.randn(1, 512, 50, 50)
x2 = torch.randn(1, 32, 50, 50)
