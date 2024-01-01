
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(2, 62, 5, stride=1, padding=10)
    def forward(self, x358):
        v1 = self.conv(x358)
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
x358 = torch.randn(1, 2, 47, 36)
