
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(1, 8, 7, stride=3, padding=7)
        self.gelu = torch.nn.GELU()
        self.conv_transpose = torch.nn.ConvTranspose2d(1, 1, (1, 1), stride=(1, 1))
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = self.gelu(v1)
        v3 = v2 * 0.5
        v4 = v2 * v2 * v2
        v5 = v4 * 0.044715
        v6 = v2 + v5
        v7 = v6 * 0.7978845608028654
        v8 = torch.tanh(v7)
        v9 = v8 + 1
        v10 = v3 * v9
        v11 = self.conv_transpose(v10)
        return v11
# Inputs to the model
x1 = torch.randn(1, 1, 35, 35)
