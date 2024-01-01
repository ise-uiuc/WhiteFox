
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(10, 30, 3, stride=1, padding=2)
    def forward(self, x7756):
        v1 = self.conv(x7756)
        v2 = v1 * torch.sin(x7756)
        v3 = v2 * 0.5
        v4 = v2 * v2
        v5 = v4 * v2
        v6 = v5 * 0.044715
        v7 = v2 + v6
        v8 = v7 * 0.7978845608028654
        v9 = torch.tanh(v8)
        v10 = v9 + 1
        v11 = v3 * v10
        return v11
# Inputs to the model
x7756 = torch.randn(1, 10, 25, 72)
