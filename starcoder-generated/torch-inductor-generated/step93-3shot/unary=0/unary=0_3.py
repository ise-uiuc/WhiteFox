
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(10, 20, 1, stride=2, padding=1)
        self.conv2 = torch.nn.Conv2d(20, 30, 3, stride=1, padding=2)
    def forward(self, x950):
        v1 = self.conv1(x950)
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
x950 = torch.randn(1, 10, 25, 65)
