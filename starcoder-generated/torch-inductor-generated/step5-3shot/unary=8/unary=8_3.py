
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose = torch.nn.ConvTranspose1d(3, 8, 3, stride=1, padding=1)
    def forward(self, x1):
        v1 = self.conv_transpose(x1)
        v2 = v1 * 2
        v3 = v2 + 3
        v4 = v3 / 2
        v5 = v4 + 3
        v6 = v5 / 2
        return v6
# Inputs to the model
x1 = torch.randn(3, 8, 64)
