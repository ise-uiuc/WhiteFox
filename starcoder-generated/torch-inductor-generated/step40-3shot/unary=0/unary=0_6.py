
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(4, 4, 3, stride=1)
        self.conv2 = torch.nn.Conv2d(4, 4, 3, stride=1)
        self.conv3 = torch.nn.Conv2d(4, 4, 3, stride=1)
        self.conv4 = torch.nn.Conv2d(4, 4, 3, stride=1)
        self.conv5 = torch.nn.Conv2d(4, 4, 3, stride=1)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = self.conv2(v1)
        v3 = self.conv3(v2)
        v4 = self.conv4(v3)
        v5 = self.conv5(v4)
        v6 = v5 * v5
        v7 = v1 + v6
        v8 = v7 * 0.7978845608028654
        v9 = torch.tanh(v8)
        v10 = v9 + 1
        v11 = v1 * v10
        return v11
# Inputs to the model
x1 = torch.randn(1, 4, 256, 256)
