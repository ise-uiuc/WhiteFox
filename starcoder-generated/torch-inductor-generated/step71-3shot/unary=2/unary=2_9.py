
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose = torch.nn.ConvTranspose2d(1, 40, (1, 3), stride=(4, 13), padding=0)
        self.softplus = torch.nn.Softplus(beta=0.5)
    def forward(self, x1):
        v1 = self.conv_transpose(x1)
        v2 = v1 * 0.5
        v3 = v1 * v1 * v1
        v4 = v3 * 0.044715
        v5 = v1 + v4
        v6 = v5 * 0.7978845608028654
        v7 = torch.tanh(v6)
        v8 = v7 * v8
        v9 = v8 * 0.7978845608028654
        v10 = torch.tanh(v9)
        v11 = v10 + v8
        v12 = v7 * v11
        v13 = self.softplus(v12)
        v14 = v13 * 1.0854228461221313
        return v14
# Inputs to the model
x1 = torch.randn(3, 1, 20, 8)
