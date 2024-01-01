
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 4, 1, stride=1)
        self.conv2 = torch.nn.Conv2d(3, 4, 1, stride=1)
        self.conv3 = torch.nn.Conv2d(3, 4, 1, stride=1)
        self.conv4 = torch.nn.Conv2d(3, 4, 1, stride=1)
        self.conv5 = torch.nn.Conv2d(3, 4, 1, stride=1)
        self.conv6 = torch.nn.Conv2d(3, 4, 1, stride=1)
        self.conv7 = torch.nn.Conv2d(3, 4, 1, stride=1)
        self.conv8 = torch.nn.Conv2d(3, 4, 1, stride=1)
    def forward(self, x1, x2):
        v1 = self.conv1(x1)
        v2 = self.conv2(x2)
        v3 = v1 + v2
        v4 = self.conv3(v3)
        v5 = self.conv4(v3)
        v6 = v4 + v5
        v7 = self.conv5(v6)
        v8 = self.conv6(v6)
        v9 = v7 + v8
        v10 = self.conv7(v9)
        v11 = self.conv8(v9)
        v12 = v10 + v11
        return v12
# Inputs to the model
x1 = torch.randn(1, 3, 128, 128)
x2 = torch.randn(1, 3, 128, 128)
