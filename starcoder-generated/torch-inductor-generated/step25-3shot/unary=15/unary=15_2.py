
class Encoder1(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 32, 3, stride=2, padding=1)
        self.conv2 = torch.nn.Conv2d(32, 32, 3, stride=1, padding=1)
        self.conv3 = torch.nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.conv4 = torch.nn.Conv2d(32, 32, 3, stride=1, padding=1)
        self.conv5 = torch.nn.Conv2d(32, 32, 3, stride=1, padding=1)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v1_out = torch.relu(v1)
        v2 = self.conv4(v1_out)
        v3 = self.conv3(v2)
        v4 = self.conv5(v2)
        v5 = torch.cat((v3, v4), 1)
        return v5
class Decoder1(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(32, 32, 3, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(32, 32, 3, stride=1, padding=1)
        self.conv3 = torch.nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.conv4 = torch.nn.Conv2d(32, 32, 3, stride=1, padding=1)
        self.conv5 = torch.nn.Conv2d(32, 32, 3, stride=1, padding=1)
        self.conv6 = torch.nn.Conv2d(64, 3, 3, stride=1, padding=1)
    def forward(self, x1):
        v6 = x1
        v7 = self.conv3(v6)
        v8 = self.conv4(v7)
        v9 = self.conv5(v6)
        v10 = torch.cat((v8, v9), 1)
        v10_out = torch.relu(v10)
        v11 = self.conv1(v10_out)
        v12 = self.conv2(v11)
        v13 = self.conv6(v12)
        return v13
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder1 = Encoder1()
        self.decoder1 = Decoder1()
    def forward(self, x1):
        v14 = self.encoder1(x1)
        v15 = self.decoder1(v14)
        return v15
# Inputs to the model
x1 = torch.randn(2, 1, 256, 256)
