
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 16, (3, 3), stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(16,16, (3, 3), stride=1, padding=0)
        self.conv3 = torch.nn.Conv2d(16, 8, (1, 1), stride=1, padding=0)
        self.conv4 = torch.nn.Conv2d(8, 16, (3, 3), stride=1)
        self.avg2d = torch.nn.AvgPool2d((2,2), stride=2, padding=0)
        self.conv5 = torch.nn.Conv2d(16,8,(2,2), stride=1, padding=0)
    def forward(self, x1):
        #v1
        v1 = self.conv1(x1)
        v2 = torch.sigmoid(v1)
        v3 = self.conv2(v2)
        v4 = torch.sigmoid(v3)
        v5 = self.conv3(v4)
        v6 = torch.sigmoid(v5)
        v7 = self.conv4(v6)
        #v2
        v8 = self.avg2d(v7)
        v9 = self.conv5(v8)
        v10 = self.sigmoid(v9)
        return v10
x1 = torch.randn(1, 3, 84, 84)
