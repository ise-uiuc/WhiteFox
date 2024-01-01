
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose_46 = torch.nn.ConvTranspose2d(1613, 6, 12, stride=1, padding=6, dilation=1)
    def forward(self, x1):
        v1 = self.conv_transpose_46(x1)
        v2 = torch.sigmoid(v1)
        v3 = v1 * v2
        return v3
# Inputs to the model
x1 = torch.randn(1, 1613, 72, 72)
