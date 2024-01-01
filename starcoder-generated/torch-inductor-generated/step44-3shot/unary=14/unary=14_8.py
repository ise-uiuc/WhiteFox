
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose_11 = torch.nn.ConvTranspose2d(21, 34, 7, stride=2, padding=3, dilation=1)
        self.conv_2 = torch.nn.Conv2d(34, 2, 7, stride=2, padding=3, dilation=1)
    def forward(self, x1):
        v1 = self.conv_transpose_11(x1)
        v2 = torch.sigmoid(v1)
        v3 = v1 * v2
        v4 = self.conv_2(v3)
        return v4
# Inputs to the model
x1 = torch.randn(1, 21, 104, 104)
