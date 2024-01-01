
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(12, 16, 2, stride=1, padding=1)
        self.conv_transpose = torch.nn.ConvTranspose2d(16, 8, 3, stride=2, padding=1, output_padding=1, groups=5)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = self.conv_transpose(v1)
        v3 = v2 * 0.5
        v4 = v2 * v2 * v2
        v5 = v4 * 0.044715
        v6 = v2 + v5
        v7 = v6 * 0.7978845608028654
        v8 = torch.tanh(v7)
        v9 = v8 + 1
        v10 = v3 * v9
        return v10
# Inputs to the model
x1 = torch.randn(33, 12, 4, 5)
