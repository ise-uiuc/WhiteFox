
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose_5 = torch.nn.ConvTranspose3d(30, 70, 3, stride=2, padding=1, groups=1)
    def forward(self, x1):
        v1 = self.conv_transpose_5(x1)
        v2 = torch.sigmoid(v1)
        v3 = v1 * v2
        return v3
# Inputs to the model
x1 = torch.randn(1, 30, 2, 28, 35)
