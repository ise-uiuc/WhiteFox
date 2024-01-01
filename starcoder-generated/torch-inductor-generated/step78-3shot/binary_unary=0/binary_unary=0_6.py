
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(16, 16, 7, stride=1, padding=3)
        self.conv_3 = torch.nn.Conv2d(16, 16, 7, stride=1, padding=3)
        self.conv_4 = torch.nn.Conv2d(16, 16, 7, stride=1, padding=3)
        self.conv_5 = torch.nn.Conv2d(16, 16, 7, stride=1, padding=3)
        self.conv_6 = torch.nn.Conv2d(16, 16, 7, stride=1, padding=3)
        self.conv_7 = torch.nn.Conv2d(16, 16, 7, stride=1, padding=3)
    def forward(self, x):
        v1 = self.conv(x)
        v2 = self.conv_3(v1)
        v3 = self.conv_4(v1)
        v4 = v1 + v3
        v5 = torch.relu(v4)
        v6 = self.conv_5(v5)
        v7 = self.conv_6(x)
        v8 = v1 + v7
        v9 = torch.relu(v8)
        v10 = self.conv_7(v9)
        v11 = v6 + v10
        v12 = torch.relu(v11)
        return v12
# Inputs to the model
x = torch.randn(1, 16, 64, 64)
