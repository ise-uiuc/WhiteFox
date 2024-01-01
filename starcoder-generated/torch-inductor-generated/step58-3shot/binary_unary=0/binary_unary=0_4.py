
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(16, 16, 7, stride=1, padding=3)
        self.conv_3 = torch.nn.Conv2d(16, 16, 7, stride=1, padding=3)
        self.conv_5 = torch.nn.Conv2d(16, 16, 7, stride=1, padding=3)
        self.conv_8 = torch.nn.Conv2d(16, 16, 7, stride=1, padding=3)
        self.conv_9 = torch.nn.Conv2d(16, 16, 15, stride=1, padding=7)
        self.conv_11 = torch.nn.Conv2d(16, 16, 7, stride=1, padding=3)
        self.conv_12 = torch.nn.Conv2d(16, 16, 7, stride=1, padding=3)
        self.conv_13 = torch.nn.Conv2d(16, 16, 7, stride=1, padding=3)
        self.conv_23 = torch.nn.Conv2d(16, 16, 7, stride=1, padding=3)
    def forward(self, x):
        v1 = self.conv(x)
        v2 = self.conv_3(v1)
        v3 = self.conv_5(v2)
        v4 = v3 + x
        v5 = torch.relu(v4)
        v6 = self.conv_8(v5)
        v7 = v6 + x
        v8 = torch.relu(v7)
        v9 = self.conv_9(v8)
        v10 = v9 + v5
        v11 = torch.relu(v10)
        v12 = self.conv_11(v11)
        v13 = v12 + v8
        v14 = torch.relu(v13)
        v15 = self.conv_12(v14)
        v16 = v15 + v6
        v17 = torch.relu(v16)
        v18 = self.conv_13(x)
        v19 = v17 + v18
        v20 = torch.relu(v19)
        v_100 = self.conv_23(v20)
        return v_100
# Inputs to the model
x = torch.randn(1, 16, 64, 64)
