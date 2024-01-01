
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(9, 19, 1, stride=3, padding=20)
        self.conv2 = torch.nn.Conv2d(19, 123, 1, stride=7, padding=16)
    def forward(self, x113):
        v1 = self.conv1(x113)
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
x113 = torch.randn(1, 9, 74, 84)
