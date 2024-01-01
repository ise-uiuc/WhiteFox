
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose2 = torch.nn.ConvTranspose2d(3, 4, 9, 2, 1, bias=True)
        self.conv_transpose = torch.nn.ConvTranspose2d(4, 9, 7, 2, 2, bias=True)
    def forward(self, x1):
        v1 = self.conv_transpose2(x1)
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
x1 = torch.randn(1, 3, 12, 15)
