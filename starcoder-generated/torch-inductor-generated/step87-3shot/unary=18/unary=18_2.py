
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(128, 64, 3, padding=1)
        self.conv2 = torch.nn.Conv2d(64, 32, 3, padding=1)
        self.conv3 = torch.nn.Conv2d(32, 32, 1)
        self.upconv = torch.nn.ConvTranspose2d(32, 32, 13)
        self.conv4 = torch.nn.Conv2d(32, 32, 13)
        self.conv5 = torch.nn.Conv2d(32, 64, 3, padding=1)
        self.conv6 = torch.nn.Conv2d(64, 128, 3, padding=1)
        self.conv7 = torch.nn.Conv2d(128, 32, 3, padding=1)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = torch.nn.functional.sigmoid(v1)
        v3 = self.conv2(v1)
        v4 = torch.nn.functional.sigmoid(v3)
        v5 = self.conv3(v4)
        v6 = torch.nn.functional.relu(v5)
        v7 = self.upconv(v6)
        v8 = torch.nn.functional.relu(v5)
        v9 = self.conv4(v8)
        v10 = torch.nn.functional.sigmoid(v9)
        v11 = self.conv5(v10)
        v12 = torch.nn.functional.relu(v11)
        v13 = self.conv6(v12)
        v14 = torch.nn.functional.relu(v13)
        v15 = self.conv7(v14)
        v16 = torch.nn.functional.sigmoid(v15)
        return v16
# Model begins
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(4, 2, bias=True)
    def forward(self, x1):
        x1 = self.linear1(x1)
        return x1
# Inputs to the model
x1 = torch.randn(1, 4)
