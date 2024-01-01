
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(5, 31, 3, stride=59, padding=13)
        self.conv2 = torch.nn.Conv2d(31, 99, 3, stride=44, padding=93)
    def forward(self, x61):
        v1 = self.conv1(x61)
        v2 = v1 * 0.5
        v3 = v1 * v1
        v4 = v3 * v1
        v5 = v4 * 0.044715
        v6 = v1 + v5
        v7 = v6 * 0.7978845608028654
        v8 = torch.tanh(v7)
        v9 = v8 + 1
        v10 = v2 * v9
        return self.conv2(v10)
# Inputs to the model
x61 = torch.randn(1, 5, 85, 5)
