
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(4, 10, 6, stride=1, padding=1, dilation=2)
        self.conv2 = torch.nn.Conv2d(10, 15, 5, stride=2, padding=1, dilation=2)
        self.conv3 = torch.nn.Conv2d(15, 15, 6, padding=1, dilation=1, groups=15)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = self.conv2(v1)
        v3 = self.conv3(v1)
        v4 = v2 + v3
        v5 = v1 + v4
        v6 = v5 * 0.5
        v7 = v5 * v5
        v8 = v7 * v5
        v9 = v8 * 0.044715
        v10 = v5 + v9
        v11 = v10 * 0.7978845608028654
        v12 = torch.tanh(v11)
        v13 = v12 + 1
        v14 = v6 * v13
        return v14
# Inputs to the model
x1 = torch.randn(1, 4, 25, 25)
