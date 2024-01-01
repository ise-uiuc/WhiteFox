
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 5, 5, stride=5, padding=4)
        self.conv2 = torch.nn.Conv2d(5, 7, 5, stride=5, padding=4)
        self.conv3 = torch.nn.Conv2d(7, 11, 5, stride=5, padding=4)
    def forward(self, x4):
        v1 = self.conv1(x4)
        v2 = self.conv2(v1)
        v3 = self.conv3(v2)
        v4 = v3 * 0.5
        v5 = v3 * v3
        v6 = v5 * v3
        v7 = v6 * 0.044715
        v8 = v3 + v7
        v9 = v8 * 0.7978845608028654
        v10 = torch.tanh(v9)
        v11 = v10 + 1
        v12 = v4 * v11
        return v12
# Inputs to the model
x4 = torch.randn(1, 1, 32, 32)
