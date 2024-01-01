
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose_1 = torch.nn.ConvTranspose2d(32, 32, 3, stride=1)
        self.conv_transpose_2 = torch.nn.ConvTranspose2d(32, 32, 3, stride=1)
        self.conv_transpose_3 = torch.nn.ConvTranspose2d(32, 3, 3, stride=1)
    def forward(self, x1):
        v1 = self.conv_transpose_1(x1)
        v2 = self.conv_transpose_2(v1)
        v3 = v2 + 3
        v4 = torch.clamp(v3, min=0)
        v5 = torch.clamp(v4, max=6)
        v6 = v2 * v5
        v7 = v2 / 6
        v8 = self.conv_transpose_3(v7)
        return v8
# Inputs to the model
x1 = torch.randn(1, 32, 60, 64)
