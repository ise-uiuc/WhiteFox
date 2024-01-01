
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose = torch.nn.ConvTranspose2d(11, 10, 1, stride=1, padding=1, bias=False)
        self.conv_transpose_0 = torch.nn.ConvTranspose2d(15, 5, 1, stride=1, padding=1, bias=False)
    def forward(self, x5):
        v1 = self.conv_transpose(x5)
        v2 = v1 * 0.5
        v3 = v1 * v1 * v1
        v4 = v3 * 0.044715
        v5 = v1 + v4
        v6 = v5 * 0.7978845608028654
        v7 = torch.tanh(v6)
        v8 = v7 + 1
        v9 = v2 * v8
        v10 = self.conv_transpose_0(torch.tanh(v9))
        return v10
# Inputs to the model
x5 = torch.randn(1, 11, 9, 4)
