
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose_1 = torch.nn.ConvTranspose2d(128, 128, 4, stride=4, padding = 0, dilation=1)
        self.conv_transpose_2 = torch.nn.ConvTranspose2d(128, 64, 4, stride=2, padding = 1, dilation=2)
        self.conv_transpose_3 = torch.nn.ConvTranspose2d(64, 32, 4, stride=2, padding = 1, dilation=2)
        self.conv_transpose_4 = torch.nn.ConvTranspose2d(32, 1, 3, stride=1, padding = 1, dilation=1)
    def forward(self, x1):
        v1 = self.conv_transpose_1(x1)
        v2 = self.conv_transpose_2(v1)
        v3 = self.conv_transpose_3(v2)
        v3 = torch.tanh(v3)
        v4 = self.conv_transpose_4(v3)
        return v4
# Inputs to the model
x1 = torch.randn(1, 128, 5, 5)
