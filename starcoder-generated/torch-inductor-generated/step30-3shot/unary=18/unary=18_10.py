
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(32, 64, kernel_size=3, stride=2)
        self.conv2 = torch.nn.Conv2d(64, 128, kernel_size=1)
        self.conv3 = torch.nn.Conv2d(128, 2, kernel_size=2, stride=1, padding=0)
        self.conv4 = torch.nn.Conv2d(32, 2, kernel_size=12, stride=4, padding=(1, 0))
        self.conv5 = torch.nn.Conv2d(2, 2, kernel_size=(1, 2), stride=5, padding=(0, 3))
        self.conv6 = torch.nn.Conv2d(2, 2, kernel_size=3, padding=1)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = self.conv1(v1)
        v3 = self.conv3(v2)
        v4 = torch.sigmoid(v1)
        v5 = torch.sigmoid(v3)
        v6 = torch.sigmoid(v4)
        v7 = torch.sigmoid(v5)
        v8 = torch.sigmoid(v6)
        v9 = torch.sigmoid(v7)
        v10 = torch.sigmoid(v8)
        v11 = self.conv4(v9)
        v12 = torch.sigmoid(v10)
        v13 = self.conv2(v11)
        v14 = self.conv5(v13)
        v15 = torch.sigmoid(v12)
        v16 = self.conv6(v14)
        v17 = nn.Sigmoid()(v16)
        return v17
# Inputs to the model
x1 = torch.randn(1, 32, 152, 212)
