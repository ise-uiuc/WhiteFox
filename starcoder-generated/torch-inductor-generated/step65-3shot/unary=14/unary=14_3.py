
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose_2 = torch.nn.ConvTranspose2d(3, 3, 4, stride=1, padding=2, groups=1, dilation=2, bias=True)
    def forward(self, x1):
        v1 = self.conv_transpose_2(x1)
        v2 = torch.sigmoid(v1)
        v3 = v1 * v2
        return v3
# Inputs to the model
x1 = torch.randn(1, 3, 48, 48)
