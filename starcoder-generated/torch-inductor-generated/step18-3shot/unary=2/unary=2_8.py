
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose = torch.nn.ConvTranspose2d(3, 2, 2, stride=2, padding=(1, 0))
        self.identity = torch.nn.ConvTranspose2d(3, 2, 2, stride=2, padding=(1, 0))
    def forward(self, x1):
        v1 = self.conv_transpose(x1)
        v2 = self.identity(x1)
        v3 = v1 * 0.5
        v4 = v1 * v1 * v1
        v5 = v4 * 0.044715
        v6 = v1 + v5
        v7 = v6 * 0.7978845608028654
        v8 = torch.tanh(v7)
        v9 = v8 + 1
        v10 = v3 * v8
        return v10
# Inputs to the model
x1 = torch.randn(2, 3, 8, 8)
