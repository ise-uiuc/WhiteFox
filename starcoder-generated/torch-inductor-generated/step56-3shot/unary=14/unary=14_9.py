
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose_0 = torch.nn.ConvTranspose2d(3022, 48, 7, stride=1, padding=3, dilation=1)
    def forward(self, x1):
        v1 = self.conv_transpose_0(x1)
        v2 = torch.sigmoid(v1)
        v3 = v1 * v2
        v4 = torch.tanh(v3)
        return v4
# Inputs to the model
x1 = torch.randn(1, 3022, 112, 112)
