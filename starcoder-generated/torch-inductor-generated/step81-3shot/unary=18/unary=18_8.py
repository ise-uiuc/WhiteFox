
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 16, (2, 2), stride=1, padding=0)
        self.conv2 = torch.nn.Conv2d(16, 16, (3, 3), stride=1, padding=0)
        self.conv3 = torch.nn.Conv2d(16, 16, (7, 7), stride=1, padding=0)
        self.conv4 = torch.nn.Conv2d(16, 16, (2, 2), stride=1, padding=0)
        self.conv5 = torch.nn.Conv2d(16, 16, (3, 3), stride=1, padding=0)
        self.conv6 = torch.nn.Conv2d(16, 16, (7, 7), stride=1, padding=0)
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
        v10 = torch.sigmoid(v9)
        v11 = self.conv6(v10)
        v12 = torch.sigmoid(v11)
        return v12
# Inputs to the model
x1 = torch.randn(1, 3, 28, 28)
