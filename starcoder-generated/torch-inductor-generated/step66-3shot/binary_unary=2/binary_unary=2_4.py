
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_1 = torch.nn.Conv2d(1, 256, 18, stride=1, padding=9)
        self.conv_2 = torch.nn.Conv2d(256, 16, 10, stride=1, padding=4)
        self.conv_3 = torch.nn.Conv2d(16, 16, 8, stride=1, padding=2)
        self.conv_4 = torch.nn.Conv2d(16, 16, 6, stride=1, padding=8)
        self.conv_5 = torch.nn.Conv2d(16, 16, 4, stride=1, padding=5)
    def forward(self, x1):
        v1 = self.conv_1(x1)
        v2 = v1 - 1
        v3 = F.relu(v2)
        v4 = self.conv_2(v3)
        v5 = v4 - 22
        v6 = F.relu(v5)
        v7 = self.conv_3(v6)
        v8 = v7 - 42.25
        v9 = F.relu(v8)
        v10 = self.conv_4(v9)
        v11 = v10 - 25
        v12 = F.relu(v11)
        v13 = self.conv_5(v12)
        v14 = v13 - 540
        v15 = F.relu(v14)
        return v15
# Inputs to the model
x1 = torch.randn(1, 1, 128, 128)
