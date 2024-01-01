
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 64, (1, 1), stride=(1, 1))
        self.conv2 = torch.nn.Conv2d(64, 128, (3, 3), stride=1, padding=1)
        self.conv3 = torch.nn.Conv2d(128, 256, (1, 1), stride=1, groups=2)
        self.conv4 = torch.nn.Conv2d(256, 256, (7, 1), stride=(1, 1), padding=(3, 0))
        self.conv5 = torch.nn.Conv2d(256, 512, (1, 7), stride=1, padding=(0, 3))
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
        return v10
# Inputs to the model
x1 = torch.randn(1, 1, 224, 224)
