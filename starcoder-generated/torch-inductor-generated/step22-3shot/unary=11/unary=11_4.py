
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose_1 = torch.nn.ConvTranspose2d(3, 8, 10, stride=1, padding=5)
        self.conv_transpose_2 = torch.nn.ConvTranspose2d(8, 6, 15, stride=1, padding=2)
        self.conv_transpose_3 = torch.nn.ConvTranspose2d(6, 4, 15, stride=1, padding=2)
        self.conv_transpose_4 = torch.nn.ConvTranspose2d(4, 2, 10, stride=1, padding=5)
    def forward(self, x1):
        v1 = self.conv_transpose_1(x1)
        v2 = self.conv_transpose_2(v1)
        v3 = self.conv_transpose_3(v2)
        v4 = self.conv_transpose_4(v3)
        v5 = v4 + 3
        v6 = torch.clamp_min(v5, 0)
        v7 = torch.clamp_max(v6, 6)
        v8 = v7 / 6
        return v8
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
