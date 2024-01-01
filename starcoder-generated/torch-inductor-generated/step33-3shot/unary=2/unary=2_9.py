
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose = torch.nn.ConvTranspose2d(1, 6, 4, stride=3, padding=2, groups=1, bias=True)
        self.relu = torch.nn.ReLU()
        self.hardswish = torch.nn.Hardswish()
        self.elu = torch.nn.ELU(0.75)
        self.selu = torch.nn.SELU(0.75)
        self.gelu = torch.nn.GELU()
    def forward(self, x1):
        v1 = self.conv_transpose(x1)
        v2 = torch.tanh(v1)
        v3 = self.relu(v1)
        v4 = self.hardswish(v1)
        v5 = self.elu(v1)
        v6 = self.selu(v1)
        v7 = self.gelu(v1)
        v8 = v2 + v3
        v9 = v4 + v5
        v10 = v6 + v7
        v11 = v8 + v9
        v12 = v10 + v11
        return v10
# Inputs to the model
x1 = torch.randn(2, 1, 3, 3)
