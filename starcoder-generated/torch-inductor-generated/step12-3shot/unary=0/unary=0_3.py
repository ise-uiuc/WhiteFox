
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(1, 1, 1, stride=2, padding=20, bias=False)
        self.conv1 = torch.nn.Conv2d(1, 1, 1, stride=2, padding=10, groups=1, bias=True)

        self.conv2 = torch.nn.Conv2d(1, 1, 3, stride=2, padding=2, groups=1, bias=True)
        self.conv3 = torch.nn.Conv2d(1, 2, 3, stride=2, padding=2, groups=1, bias=True)
        self.conv4 = torch.nn.Conv2d(1, 3, 3, stride=2, padding=2, groups=1, bias=True)
        self.conv5 = torch.nn.Conv2d(1, 4, 3, stride=2, padding=2, groups=1, bias=True)
    def forward(self, x4):
        v1 = self.conv(x4)
        v2 = self.conv1(x4)

        v3 = self.conv2(x4)
        v4 = self.conv3(x4)
        v5 = self.conv4(x4)
        v6 = self.conv1(v5)
        v7 = v1 + v3 + v4 + v6

        v8 = self.conv1(v7)
        v9 = v2 + v7 + v8

        v10 = v9 * 0.5
        v11 = v9 * v9
        v12 = v11 * v9
        v13 = v12 * 0.044715
        v14 = v9 + v13
        v15 = v14 * 0.7978845608028654
        v16 = torch.tanh(v15)
        v17 = v16 + 1
        v18 = v10 * v17
        return v18
# Inputs to the model
x4 = torch.randn(1, 1, 80, 80)
