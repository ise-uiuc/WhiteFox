
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose_125 = torch.nn.ConvTranspose2d(34, 25, 1, stride=1, padding=0)
    def forward(self, x1):
        v1 = self.conv_transpose_125(x1)
        v2 = torch.sigmoid(v1)
        v3 = v1 * v2
        return v3
# Inputs to the model
x1 = torch.randn(1, 125, 64, 64)
