
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(16, 16, 7, stride=1, padding=3)
        self.conv2 = torch.nn.Conv2d(16, 16, 7, stride=1, padding=3)
        self.conv3 = torch.nn.Conv2d(16, 16, 7, stride=1, padding=3)
        self.conv4 = torch.nn.Conv2d(16, 16, 7, stride=1, padding=3)
    def forward(self, x1, x1_2, x2, x2_2):
        v1 = self.conv1(x1)
        v2 = self.conv2(x2)
        v3 = v1 + v2
        v4 = torch.relu(v3)
        v5 = self.conv3(v4)
        v6 = v5 + x1
        v7 = torch.relu(v6)
        v8 = self.conv4(v7)
        v9 = torch.relu(v8)
        v10 = v9 + x1_2
        v11 = v10 + x2_2
        return v11
# Inputs to the model
x1 = torch.randn(1, 16, 64, 64)
x1_2 = torch.randn(1, 16, 64, 64)
x2 = torch.randn(1, 16, 64, 64)
x2_2 = torch.randn(1, 16, 64, 64)
