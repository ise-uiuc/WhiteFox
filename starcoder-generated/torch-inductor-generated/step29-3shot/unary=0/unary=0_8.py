
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(2, 4, 3, stride=3, padding=1)
        self.conv2 = torch.nn.Conv2d(4, 4, 3, stride=3, padding=1)
    def forward(self, x4):
        v1 = self.conv1(x4)
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
x4 = torch.randn(1, 2, 36, 72)
