
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(16, 16, 7, stride=1, padding=3)
        self.conv2 = torch.nn.Conv2d(16, 16, 7, stride=1, padding=3)
        self.conv3 = torch.nn.Conv2d(16, 16, 7, stride=1, padding=3)
    def forward(self, x9_1, x3, x7_1, x3_2, x3_1, x4_1, x3_3, x7_2):
        v1 = self.conv1(x3)
        v2 = self.conv2(x9_1)
        a1 = self.conv1(x7_2)
        v3 = v1 + x3
        v4 = self.conv3(x3_2)
        a2 = self.conv3(x3_3)
        a3 = self.conv2(x3_1)
        v5 = v3 + x4_1
        v6 = torch.relu(v5)
        a4 = a2 + x7_1
        v7 = a3 + x3
        v8 = torch.relu(v7)
        a5 = self.conv3(v8)
        v9 = v4 + v6
        a6 = self.conv3(v9)
        v10 = torch.relu(a4)
        a7 = a1 + a6
        v11 = a5 + v10
        v12 = torch.relu(a7)
        return v12
# Inputs to the model
x9_1 = torch.randn(1, 16, 64, 64)
x3 = torch.randn(1, 16, 64, 64)
x7_1 = torch.randn(1, 16, 64, 64)
x3_2 = torch.randn(1, 16, 64, 64)
x3_1 = torch.randn(1, 16, 64, 64)
x4_1 = torch.randn(1, 16, 64, 64)
x3_3 = torch.randn(1, 16, 64, 64)
x7_2 = torch.randn(1, 16, 64, 64)
