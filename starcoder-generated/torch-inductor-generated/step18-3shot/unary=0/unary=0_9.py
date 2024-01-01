
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 1, 2, stride=2, padding=2, dilation=2, groups=1)
        self.conv1 = torch.nn.Conv2d(3, 3, 2, stride=2, padding=2, dilation=2, groups=3)
        self.conv2 = torch.nn.Conv2d(3, 3, 2, stride=2, padding=2, dilation=2, groups=3)
        self.conv3 = torch.nn.Conv2d(3, 3, 2, stride=3, padding=1, dilation=1, groups=3)
    def forward(self, x0):
        v0 = self.conv(x0)
        v1 = self.conv1(x0)
        v2 = self.conv2(x0)
        v3 = self.conv3(x0)
        v4 = v0 * 0.5
        v5 = v1 * v1
        v6 = v2 * v2
        v7 = v3 * v3
        v8 = v4 * 0.044715
        v9 = v5 * 0.044715
        v10 = v1 + v9
        v11 = v10 * 0.7978845608028654
        v12 = torch.tanh(v11)
        v13 = v12 + 1
        v14 = v3 * v13
        v15 = v6 * 0.044715
        v16 = v7 * 0.044715
        v17 = v15 + v16
        v18 = v17 * 0.7978845608028654
        v19 = torch.tanh(v18)
        v20 = v19 + 1
        v21 = v2 * v20
        return v21
# Inputs to the model
x0 = torch.randn(1, 3, 50, 50)
