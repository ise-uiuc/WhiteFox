
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose_3 = torch.nn.ConvTranspose2d(16, 16, 5, stride=1, padding=0, groups=1, dilation=2, bias=False)
    def forward(self, x1):
        v1 = self.conv_transpose_3(x1)
        v2 = torch.sigmoid(v1)
        v3 = v1 * v2
        return v3
# Inputs to the model
x1 = torch.randn(2, 16, 28, 28)
