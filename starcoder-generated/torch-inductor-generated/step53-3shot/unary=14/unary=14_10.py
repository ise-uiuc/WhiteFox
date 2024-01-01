
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose_33 = torch.nn.ConvTranspose2d(16, 44, 3, stride=2, padding=1, dilation=1)
        self.conv_transpose_35 = torch.nn.ConvTranspose2d(44, 44, 1, stride=2, padding=0, dilation=1)
    def forward(self, x1):
        v1 = self.conv_transpose_33(x1)
        v2 = torch.sigmoid(v1)
        v3 = v1 * v2
        v4 = self.conv_transpose_35(v3)
        return v4
# Inputs to the model
x1 = torch.randn(1, 16, 19, 19)
