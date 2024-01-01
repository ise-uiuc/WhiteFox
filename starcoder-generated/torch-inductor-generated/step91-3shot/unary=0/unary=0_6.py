
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(63, 123, 3, stride=1, padding=2)
    def forward(self, x8):
        v2 = torch.randn(3, 101, 69, 87)
        v1 = self.conv(x8)
        v3 = v1 * 0.5
        v4 = v1 * v1
        v5 = v4 * 0.044715
        v6 = v1 + v5
        v7 = v6 * 0.7978845608028654
        v8 = torch.tanh(v7)
        v9 = v8 + 1
        v10 = v3 + v9
        v11 = v10 + torch.tanh(v7)
        v12 = v2 * v11
        return v12
# Inputs to the model
x8 = torch.randn(1, 63, 93, 45)
