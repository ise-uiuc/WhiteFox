
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose_512_0 = torch.nn.ConvTranspose2d(512, 256, 1, stride=1, padding=0)
        self.conv_transpose_512_1 = torch.nn.ConvTranspose2d(256, 256, 1, stride=1, padding=0)
    def forward(self, x1):
        v1 = self.conv_transpose_512_0(x1)
        v2 = torch.sigmoid(v1)
        v3 = v1 * v2
        v4 = self.conv_transpose_512_1(v3)
        v5 = torch.sigmoid(v4)
        v6 = v4 * v5
        return v6
# Inputs to the model
x1 = torch.randn(1, 512, 16, 16)
