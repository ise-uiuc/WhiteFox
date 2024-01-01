
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose = torch.nn.ConvTranspose2d(6, 10, kernel_size=2, stride=1, padding=0)
        self.conv_transpose2 = torch.nn.ConvTranspose2d(10, 12, kernel_size=1, stride=1, padding=0)
        self.conv_transpose3 = torch.nn.ConvTranspose2d(12, 14, kernel_size=1, stride=1, padding=0)
    def forward(self, x1):
        v1 = self.conv_transpose(x1)
        v2 = v1 * 5
        v3 = v1 * v1 * v1
        v4 = v3 * 0.5
        v5 = v1 + v4
        v6 = v5 * 0.7978845608028654
        v7 = torch.tanh(v6)
        v8 = v7 + 1
        v9 = v2 * v8
        v10 = self.conv_transpose2(v9)
        v11 = v10 * 0.04473068457651329
        v12 = self.conv_transpose3(v11)
        v13 = v12 * (-2878.5064697265625)
        return v13
# Inputs to the model
x1 = torch.randn(1, 6, 64, 64)
