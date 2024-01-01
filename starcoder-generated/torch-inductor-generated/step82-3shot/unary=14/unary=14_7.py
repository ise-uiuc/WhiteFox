
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose_256_1 = torch.nn.ConvTranspose2d(27, 18, 3, stride=1, padding=1)
        self.conv_256_2 = torch.nn.Conv2d(256, 7, 7, stride=1, padding=6)
    def forward(self, x1, x2):
        v1 = self.conv_transpose_256_1(x2)
        v2 = torch.sigmoid(v1)
        v3 = v1 * v2
        v4 = self.conv_256_2(v3)
        v5 = torch.sigmoid(v4)
        v6 = v4 * v5
        return v6
# Inputs to the model
x1 = torch.randn(1, 27, 24, 24)
x2 = torch.randn(1, 256, 64, 64)
