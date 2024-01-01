
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(2, 6, 1, stride=1, padding=1)
    def forward(self, x13):
        v8 = torch.tanh(x13)
        v15 = v8.repeat((1, 4, 1, 2))
        v20 = torch.tanh(v15)
        v16 = v8.repeat((1, 2, 2, 1))
        v17 = torch.cat([v8, v20, v16, v8, v20, v16], 0)
        v18 = torch.nn.functional.interpolate(v17, None, 8, 'bicubic', True)
        v16 = torch.nn.functional.interpolate(v16, None, 6, 'bicubic', True)
        v15 = torch.nn.functional.interpolate(v20, None, 3, 'linear', True)
        v1 = self.conv(v18)
        v2 = v1 * 0.5
        v3 = v1 * v1
        v4 = v3 * v1
        v5 = v4 * 0.044715
        v6 = v1 + v5
        v7 = v6 * 0.7978845608028654
        v8 = torch.tanh(v7)
        v9 = v8 + 1
        v10 = v2 * v9
        return v10
# Inputs to the model
x13 = torch.randn(1, 2, 1, 1)
