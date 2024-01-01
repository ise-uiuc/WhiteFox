
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose = torch.nn.ConvTranspose1d(2, 4, (3, 5), stride=(3, 5), groups=2)
    def forward(self, x1):
        v2 = self.conv_transpose(x1)
        v3 = v2 * 0.5
        v4 = v2 * v2 * v2
        v5 = v4 * 0.044715
        v6 = v2 + v5
        v7 = v6 * 0.7978845608028654
        v8 = torch.tanh(v7)
        v9 = v8 + 1
        v10 = v3 * v9
        v11 = torch.tanh(v10)
        return v11
# Inputs to the model
x1 = torch.randn(4, 2, 2)
