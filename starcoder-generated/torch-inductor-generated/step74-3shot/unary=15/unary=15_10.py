
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv11 = torch.nn.Conv2d(3, 8, (7, 3), stride=(2, 1), padding=(2, 1))
        self.conv12 = torch.nn.Conv2d(8, 16, (3, 3), stride=1, padding=(1, 1))
        self.conv13 = torch.nn.Conv2d(16, 8, (3, 3), stride=(2, 1), padding=(1, 1))
        self.conv21 = torch.nn.Conv2d(8, 8, (5, 3), stride=1, padding=(1, 1))
        self.conv22 = torch.nn.Conv2d(8, 32, (3, 1), stride=(2, 1), padding=(1, 1))
        self.conv31 = torch.nn.Conv2d(32, 8, (7, 3), stride=(2, 1), padding=(2, 1))
        self.conv32 = torch.nn.Conv2d(8, 64, (3, 3), stride=1, padding=(1, 1))
        self.conv33 = torch.nn.Conv2d(64, 32, (3, 3), stride=(2, 1), padding=(1, 1))
    def forward(self, x1):
        v1 = self.conv11(x1)
        v2 = torch.relu(v1)
        v3 = self.conv12(v2)
        v4 = torch.relu(v3)
        v5 = self.conv13(v4)
        v6 = torch.relu(v5)
        v7 = self.conv21(v6)
        v8 = torch.relu(v7)
        v9 = self.conv22(v8)
        v10 = torch.relu(v9)
        v11 = self.conv31(v10)
        v12 = torch.relu(v11)
        v13 = self.conv32(v12)
        v14 = torch.relu(v13)
        v15 = self.conv33(v14)
        v16 = torch.relu(v15)
        return v16
# Inputs to the model
x1 = torch.randn(1, 3, 256, 416)
