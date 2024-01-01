
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose1 = torch.nn.ConvTranspose2d(1000, 32, 4, stride=1, padding=0, output_padding=0)
        self.conv_transpose2 = torch.nn.ConvTranspose2d(32, 32, 2, stride=1, padding=0, output_padding=0)
        self.conv_transpose3 = torch.nn.ConvTranspose2d(32, 3, 2, stride=2, padding=0, output_padding=0)
    def forward(self, x1):
        v1 = self.conv_transpose1(x1)
        v2 = v1 * 0.5
        v3 = v1 * v1 * v1
        v4 = v3 * 0.044715
        v5 = v1 + v4
        v6 = v5 * 0.7978845608028654
        v7 = torch.tanh(v6)
        v8 = v7 + 1
        v9 = v2 * v8
        v10 = self.conv_transpose2(v9)
        v11 = v10 * 0.5
        v12 = v10 * v10 * v10
        v13 = v12 * 0.044715
        v14 = v10 + v13
        v15 = v14 * 0.7978845608028654
        v16 = torch.tanh(v15)
        v17 = v16 + 1
        v18 = v11 * v17
        v19 = self.conv_transpose3(v18)
        return v19
# Inputs to the model
x1 = torch.randn(1, 32, 24, 24)
