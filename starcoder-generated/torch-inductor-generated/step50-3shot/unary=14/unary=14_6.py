
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose_4 = torch.nn.ConvTranspose2d(32, 24, 5, stride=2, padding=2)
    def forward(self, x1):
        v1 = self.conv_transpose_4(x1)
        v2 = torch.sigmoid(v1)
        v3 = v1 * v2
        return v3
# Inputs to the model
x1 = torch.randn(1, 32, 128, 32)
