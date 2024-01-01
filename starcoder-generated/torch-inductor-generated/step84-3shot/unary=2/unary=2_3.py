
class Model_1(torch.nn.Module):
    def __init__(self, v2):
        super().__init__()
        self.conv_transpose = torch.nn.ConvTranspose2d(16, 19, 7, padding=1, dilation=1, stride=5)
        self.v2 = v2
    def forward(self, x1):
        v1 = self.conv_transpose(x1)
        v4 = v1 * 0.5
        v5 = v1 * v1 * v1
        v6 = v5 * 0.044715
        v7 = v1 + v6
        v9 = v7 * 0.7978845608028654
        v3 = torch.tanh(v9)
        v10 = v3 + 1
        v11 = v4 * v10
        return v11
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.module = Model_1(torch.nn.ConvTranspose2d(16, 19, 7, padding=1, dilation=1, stride=5))
    def forward(self, x1):
        v12 = self.module(x1) + 1
        return v12
# Inputs to the model
x1 = torch.randn(1, 16, 9, 9)
