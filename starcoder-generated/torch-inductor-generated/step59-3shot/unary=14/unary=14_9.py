
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose_3 = torch.nn.ConvTranspose2d(1792, 32, 8, stride=2, padding=3, dilation=4)
    def forward(self, x1):
        v1 = self.conv_transpose_3(x1)
        v2 = torch.sigmoid(v1)
        v3 = v1 * v2
        v4 = torch.tanh(v3)
        return v4
# Inputs to the model
x1 = torch.randn(1, 1792, 80, 80)
