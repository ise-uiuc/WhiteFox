
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose_0 = torch.nn.ConvTranspose2d(216, 64, 3, stride=2, padding=1, dilation=1)
        self.conv_transpose_1 = torch.nn.ConvTranspose2d(64, 64, 3, stride=1, padding=1, dilation=1)
        self.conv_transpose_2 = torch.nn.ConvTranspose2d(64, 3, 3, stride=1, padding=1, dilation=1)
    def forward(self, x1):
        v2 = self.conv_transpose_0(x1)
        v3 = v2
        v5 = self.conv_transpose_1(v3)
        v6 = v5
        v8 = self.conv_transpose_2(v6)
        return v8
# Inputs to the model
x1 = torch.randn(1, 216, 64, 64)
