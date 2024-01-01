
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose_7 = torch.nn.ConvTranspose2d(182, 64, 7, stride=5, padding=3, dilation=2)
    def forward(self, x1):
        v1 = self.conv_transpose_7(x1)
        v2 = torch.sigmoid(v1)
        v3 = v1 * v2
        return v3
# Inputs to the model
x1 = torch.randn(1, 182, 204, 204)
