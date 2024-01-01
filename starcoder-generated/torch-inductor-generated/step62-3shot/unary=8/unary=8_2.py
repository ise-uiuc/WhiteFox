
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose = torch.nn.ConvTranspose2d(3, 32, 3, stride=1)
    def forward(self, x1):
        v1 = self.conv_transpose(x1)
        v2 = v1 + 3
        v3 = v1 - 5
        v4 = torch.clamp(v2, min=0)
        v5 = torch.clamp(v3, max=6)
        v6 = v1 * v5
        v7 = v1 / 6
        v8 = torch.tanh(v7)
        v9 = v1 + 5
        return v2
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
