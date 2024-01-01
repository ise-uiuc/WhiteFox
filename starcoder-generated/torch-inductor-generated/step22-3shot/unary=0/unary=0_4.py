
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(32, 32, 2, stride=1, padding=0)
        self.conv2 = torch.nn.Conv2d(32, 128, 3, stride=1, padding=0)
        self.conv3 = torch.nn.Conv2d(1, 128, 9, stride=1, padding=0)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = self.conv2(v1)
        v3 = self.conv3(x1)
        v4 = v2 + v3
        v5 = v4 * 0.5
        v6 = v4 * v4
        v7 = v6 * v4
        v8 = v7 * 0.044715
        v9 = v4 + v8
        v10 = v9 * 0.7978845608028654
        v11 = torch.tanh(v10)
        v12 = v11 + 1
        v13 = v5 * v12
        return v13
# Inputs to the model
x1 = torch.randn(1, 32, 32, 32)
