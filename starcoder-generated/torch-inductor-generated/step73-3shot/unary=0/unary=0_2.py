
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(2,3, 3, stride=1, padding=3)
        self.conv2 = torch.nn.Conv2d(3, 2, 1, stride=2, padding=0)
        self.conv3 = torch.nn.Conv2d(2, 3, 3, stride=1, padding=2)
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
x4 = torch.randn(1, 2, 6, 6)
