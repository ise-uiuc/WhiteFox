
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose_0 = torch.nn.ConvTranspose2d(3, 64, 7, stride=2, padding=0, dilation=1)
    def forward(self, x1):
        v1 = self.conv_transpose_0(x1)
        v2 = torch.exp(v1)
        v3 = v1 * v2
        return v3
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
