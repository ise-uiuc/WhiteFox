
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose_16 = torch.nn.ConvTranspose2d(8, 8, 15, stride=1, groups=2, padding=7)
    def forward(self, x1):
        v1 = self.conv_transpose_16(x1)
        v2 = torch.sigmoid(v1)
        v3 = v1 * v2
        return v3
# Inputs to the model
x1 = torch.randn(1, 8, 5, 6)
