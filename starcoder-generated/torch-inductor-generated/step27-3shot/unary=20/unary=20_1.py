
class Model_v2(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose_1 = torch.nn.ConvTranspose2d(32, 16, 3, stride=2, groups=2)
        self.conv_transpose_2 = torch.nn.ConvTranspose2d(16, 8, 2, stride=2, dilation=1, groups=1)
        self.conv_transpose_3 = torch.nn.ConvTranspose2d(8, 8, 2, stride=1, dilation=1, groups=1)
    def forward(self, x1):
        v0 = self.conv_transpose_1(x1)
        v1 = torch.sigmoid(v0)
        v2 = self.conv_transpose_2(v1)
        v3 = torch.tanh(v2)
        v4 = self.conv_transpose_3(v3)
        return v4

model_v2 = Model_v2()
