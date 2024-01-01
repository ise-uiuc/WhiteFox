
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose = torch.nn.ConvTranspose2d(2, 17, 1, stride=1, padding=0, bias=False)
    def forward(self, x1):
        v1 = self.conv_transpose(x1)
        v2 = v1 * 1
        v3 = v1 * v1 * v1
        v4 = v3 * 1
        v5 = v1 + v4
        v6 = v5 * 1
        v7 = torch.tanh(v6)
        v8 = v7 + 1
        v9 = v2 * v8
        v10 = v9 * 1
        return v10
# Inputs to the model
x1 = torch.randn(9, 2, 3, 5)
