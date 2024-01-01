
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose_1 = torch.nn.ConvTranspose2d(1, 8, 17, stride=4, padding=(4, 4))
        self.conv_transpose_2 = torch.nn.ConvTranspose2d(8, 8, 12, stride=2)
    def forward(self, x1):
        v1 = self.conv_transpose_1(x1)
        v2 = nn.BatchNorm2d(8, eps=2.000000)
        v3 = v2(v1)
        v4 = torch.sigmoid(v3)
        v5 = v3 * v4
        v6 = self.conv_transpose_2(v5)
        return v6
# Inputs to the model
x1 = torch.randn(1, 1, 37, 37)
