
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(54, 9, 7, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(41, 3, 1, stride=1, padding=1)
    def forward(self, x3):
        v1 = self.conv1(x3)
        v2 = v1 * 0.5
        v3 = v1 * v1
        v4 = v3 * v1
        v5 = v4 * 0.044715
        v6 = v1 + v5
        v7 = v6 * 0.7978845608028654
        v8 = torch.tanh(v7)
        v9 = v8 + 1
        v10 = v2 * v9
        v11 = self.conv2(v10)
        return v11
# Inputs to the model
x3 = torch.randn(1, 54, 64, 64)
