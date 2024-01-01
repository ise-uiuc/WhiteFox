
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(144, 56, 1, stride=1, padding=1)
        self.conv1 = torch.nn.Conv2d(28, 1024, 1, stride=1, padding=0)
        self.conv2 = torch.nn.Conv2d(14, 384, 1, stride=1, padding=0)
        self.conv3 = torch.nn.Conv2d(72, 1024, 1, stride=1, padding=0)
        self.conv4 = torch.nn.Conv2d(80, 154, 1, stride=1, padding=0)
        self.conv5 = torch.nn.Conv2d(16, 192, 1, stride=1, padding=0)
        self.conv6 = torch.nn.Conv2d(22, 192, 1, stride=1, padding=0)
        self.conv7 = torch.nn.Conv2d(40, 384, 1, stride=1, padding=0)
        self.conv8 = torch.nn.Conv2d(27, 62, 1, stride=1, padding=0)
        self.conv9 = torch.nn.Conv2d(15, 56, 1, stride=1, padding=0)
        self.conv10 = torch.nn.Conv2d(56, 77, 1, stride=1, padding=0)
    def forward(self, x3):
        v1 = self.conv(x3)
        v2 = self.conv1(x3)
        v3 = self.conv2(x3)
        v4 = self.conv3(x3)
        v5 = self.conv4(x3)
        v6 = self.conv5(x3)
        v7 = self.conv6(x3)
        v8 = self.conv7(x3)
        v9 = self.conv8(x3)
        v10 = self.conv9(x3)
        v11 = self.conv10(x3)
        v12 = v1 * 0.5
        v13 = v1 * v1
        v14 = v13 * v1
        v15 = v14 * 0.044715
        v16 = v1 + v15
        v17 = v16 * 0.7978845608028654
        v18 = torch.tanh(v17)
        v19 = v18 + 1
        v20 = v12 * v19
        return v20
# Inputs to the model
x3 = torch.randn(1, 288, 4, 4)
