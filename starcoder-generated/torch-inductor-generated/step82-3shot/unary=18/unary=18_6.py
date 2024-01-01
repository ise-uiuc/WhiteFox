
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv0 = torch.nn.Conv2d(3, 4, 1, stride=1, padding=0)
        self.conv1 = torch.nn.Conv2d(14, 7, 1, stride=1, padding=0)
        self.conv2 = torch.nn.Conv2d(7, 20, 1, stride=1, padding=0)
        self.conv3 = torch.nn.Conv2d(20, 10, 1, stride=1, padding=0)
        self.conv4 = torch.nn.Conv2d(10, 5, 1, stride=1, padding=0)
        self.conv5 = torch.nn.Conv2d(5, 19, 1, stride=1, padding=0)
        self.conv6 = torch.nn.Conv2d(19, 18, 1, stride=1, padding=0)
    def forward(self, x1):
        v1 = self.conv0(x1)
        v2 = self.conv1(v1)
        v3 = torch.sigmoid(v2)
        v4 = self.conv2(v3)
        v5 = self.conv3(v4)
        v6 = torch.sigmoid(v5)
        v7 = self.conv4(v6)
        v8 = self.conv5(v7)
        v9 = torch.sigmoid(v8)
        v10 = self.conv6(v9)
        v11 = torch.sigmoid(v10)
        return v11
# Inputs to the model
x1 = torch.randn(1, 3, 512, 256)
