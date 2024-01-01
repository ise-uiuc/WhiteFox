
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(64, 128, 1, stride=1, padding=0)
        self.conv1 = torch.nn.Conv2d(128, 256, 1, stride=1, padding=0)
        self.conv2 = torch.nn.Conv2d(256, 512, 1, stride=1, padding=0)
        self.conv3 = torch.nn.Conv2d(512, 1024, 1, stride=1, padding=0)
    def forward(self, x13):
        v1 = self.conv(x13)
        v2 = self.conv1(v1)
        v3 = self.conv2(v2)
        v4 = self.conv3(v3)
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
x13 = torch.randn(8, 64, 64, 64)
