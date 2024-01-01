
# Note that the ConvTranspose2d function is not applied to the first tensor.
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose = torch.nn.ConvTranspose2d(44, 24, 7, stride=2, padding=0)
    def forward(self, x1):
        v1 = v2 = v3 = v4 = self.conv_transpose(x1)
        v5 = v1 * 0.5
        v6 = v1 * 0.7071067811865476
        v7 = torch.erf(v6)
        v8 = v7 + 1
        v9 = v5 * v8
        v10 = self.conv_transpose(x1)
        return v9, v10
# Inputs to the model
x1 = torch.randn(5, 44, 5, 5)
