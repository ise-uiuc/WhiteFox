
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose_1 = torch.nn.ConvTranspose2d(64, 64, 3, stride=1, padding=1, dilation=1, groups=1)
        self.conv_transpose_2 = torch.nn.ConvTranspose2d(64, 64, 3, stride=1, padding=1, dilation=1, groups=1)
    def forward(self, x1):
        t1 = self.conv_transpose_1(x1)
        t2 = torch.tanh(t1)
        t3 = self.conv_transpose_2(t2)
        t4 = torch.sigmoid(t3)
        t5 = t2 + t4
        t6 = torch.sigmoid(t5)
        t7 = t5 * t6
        return t7
# Inputs to the model
x1 = torch.randn(1, 64, 31, 31)
