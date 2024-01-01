
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(26, 13, 5, stride=2, padding=1, groups=13)
        self.conv2 = torch.nn.Conv2d(6, 3, 2, stride=1, padding=0, groups=3)
    def forward(self, x6):
        v1 = self.conv1(x6)
        v2 = self.conv2(v1)
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
x6 = torch.randn(1, 26, 112, 112)
