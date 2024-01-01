
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose_1 = torch.nn.ConvTranspose2d(2, 7, 5, stride=3, padding=4, bias=True)
        self.conv_transpose_2 = torch.nn.ConvTranspose2d(5, 9, 3, stride=4, bias=False)
        self.conv_transpose_3 = torch.nn.ConvTranspose2d(1, 10, stride=(3, 3), padding=(1, 1))
    def forward(self, x1):
        v1 = self.conv_transpose_1(x1)
        v2 = self.conv_transpose_2(v1)
        v3 = v2 * 0.5
        v4 = v2 * v2 * v2
        v5 = v4 * 0.044715
        v6 = v2 + v5
        v7 = v6 * 0.7978845608028654
        v8 = torch.tanh(v7)
        v9 = v8 + 1
        v10 = v3 * v9
        v11 = self.conv_transpose_3(x1)
        return x1, v10, v11
# Inputs to the model
x1 = torch.randn(3, 2, 4, 4)
