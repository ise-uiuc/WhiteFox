
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(97, 71, 30, stride=9, padding=11)
        self.conv2 = torch.nn.Conv2d(71, 70, 20, stride=8, padding=10)
        self.conv3 = torch.nn.Conv2d(70, 49, 10, stride=7, padding=9)
        self.conv4 = torch.nn.Conv2d(49, 39, 5, stride=6, padding=4)
    def forward(self, x4):
        v1 = self.conv1(x4)
        v2 = self.conv2(v1)
        v3 = self.conv3(v2)
        v4 = self.conv4(v3)
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
x4 = torch.randn(1, 97, 61, 83)
