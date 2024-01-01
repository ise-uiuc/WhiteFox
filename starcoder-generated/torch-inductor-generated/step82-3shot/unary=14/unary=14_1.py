
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose_256_1 = torch.nn.ConvTranspose2d(128, 64, 3, stride=1, padding=1)
        self.conv_transpose_256_2 = torch.nn.ConvTranspose2d(64, 8, 5, stride=2, padding=2)
    def forward(self, x1):
        v1 = self.conv_transpose_256_1(x1)
        v2 = torch.sigmoid(v1)
        v3 = v1 * v2
        v4 = self.conv_transpose_256_2(v3)
        v5 = torch.sigmoid(v4)
        v6 = v4 * v5
        return v6
# Inputs to the model
x1 = torch.randn(1, 128, 16, 16)
