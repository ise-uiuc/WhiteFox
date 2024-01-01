
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose_3 = torch.nn.ConvTranspose2d(64, 64, 3, stride=1, padding=3, dilation=2)
    def forward(self, x1):
        v1 = self.conv_transpose_3(x1)
        v2 = (v1 * 0.05362775835652351) - 0.026444369011540413
        v3 = v1 * v2
        v4 = v1 + 0.12492471303987503
        v5 = v3 * v4
        return v5
# Inputs to the model
x1 = torch.randn(1, 64, 64, 64)
