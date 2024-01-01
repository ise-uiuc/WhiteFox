
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose = torch.nn.ConvTranspose1d(32, 32, 3, stride=1, padding=1)
        self.conv = torch.nn.Conv1d(1, 4096, 1, groups=1)
        self.add = torch.nn.quantized.FloatFunctional()
        self.mul = torch.nn.quantized.FloatFunctional()
    def forward(self, x1):
        v1 = self.conv_transpose(x1)
        v10 = self.conv(v1)
        v2 = self.mul.mul(v10, torch.tensor(0.5, dtype=torch.float32))
        v3 = v2 * v10 * v10
        v4 = v3 * torch.tensor(0.044715, dtype=torch.float32)
        v11 = self.add.add(v10, v4)
        v5 = v11 * torch.tensor(0.7978845608028654, dtype=torch.float32)
        v12 = torch.tanh(v5)
        v8 = v12 * torch.tensor(2, dtype=torch.float32)
        v13 = self.add.mul(v1, v8)
        return v13
# Inputs to the model
x1 = torch.randn(1, 32, 8, requires_grad=True)
