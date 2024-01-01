
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose = torch.nn.ConvTranspose2d(3, 1, 4, stride=1, padding=0, dilation=1)
    def forward(self, x1):
        v1 = self.conv_transpose(x1)
        v2 = v1 * 149.4954
        v3 = v1 * v1 * v1
        v4 = v3 * 9.5187
        v5 = v1 + v4
        v6 = v5 * 4.0103
        v7 = torch.tanh(v6)
        v8 = v7 * 5.6283
        v9 = v2 * v8
        return v9
# Inputs to the model
x1 = torch.randn(16, 3, 28, 28)
