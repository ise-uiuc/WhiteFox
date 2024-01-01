
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose_56 = torch.nn.ConvTranspose2d(384, 512, 3, stride=2, padding=1, dilation=2)
    def forward(self, x1):
        v1 = self.conv_transpose_56(x1)
        v2 = torch.sigmoid(v1)
        v3 = v1 * v2
        return v3
# Inputs to the model
x1 = torch.randn(1, 384, 12, 12)
