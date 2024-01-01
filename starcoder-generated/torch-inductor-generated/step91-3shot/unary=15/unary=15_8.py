
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(23, 19, 13, stride=5, padding=0)
        self.conv2 = torch.nn.Conv2d(17, 47, 13, stride=7, padding=0)
        self.conv3 = torch.nn.Conv2d(82, 47, 13, stride=15, padding=0)
        self.conv4 = torch.nn.Conv2d(85, 46, 13, stride=1, padding=0)
        self.conv5 = torch.nn.Conv2d(86, 34, 13, stride=1, padding=0)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = torch.sigmoid(v1)
        v3 = self.conv2(v2)
        v4 = torch.sigmoid(v3)
        v5 = self.conv3(v4)
        v6 = torch.sigmoid(v5)
        v7 = self.conv4(v6)
        v8 = torch.sigmoid(v7)
        v9 = self.conv5(v8)
        return v9
# Inputs to the model
x1 = torch.randn(1, 23, 32, 32)
