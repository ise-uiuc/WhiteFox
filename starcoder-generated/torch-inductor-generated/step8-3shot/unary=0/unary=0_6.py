
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(64, 8, 1, stride=1, padding=0)
        self.conv1 = torch.nn.Conv2d(8, 128, 1, stride=2, padding=0)
        self.conv2 = torch.nn.Conv2d(128, 192, 1, stride=2, padding=0)
        self.conv3 = torch.nn.Conv2d(192, 256, 3, stride=3, padding=1)
        self.conv4 = torch.nn.Conv2d(256, 384, 1, stride=2, padding=0)
    def forward(self, x3):
        v1 = self.conv(x3)
        v2 = self.conv1(v1)
        v3 = self.conv2(v2)
        v4 = self.conv3(v3)
        v5 = self.conv4(v4)
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
x3 = torch.randn(1, 64, 16, 16)
