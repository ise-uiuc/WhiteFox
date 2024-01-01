
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose = torch.nn.ConvTranspose1d(1, 2, 3, stride=2, padding=1)
    def forward(self, x6):
        v5 = self.conv_transpose(x6)
        v6 = v5 * 0.5
        v7 = v5 * v5
        v8 = v7 * 0.00022234943707104842
        v9 = v5 + v8
        v10 = v9 * 0.444715
        v11 = torch.tanh(v10)
        v12 = v11 + 1
        v13 = v6 * v12
        return v13
# Inputs to the model
x6 = torch.randn(1, 1, 6)
