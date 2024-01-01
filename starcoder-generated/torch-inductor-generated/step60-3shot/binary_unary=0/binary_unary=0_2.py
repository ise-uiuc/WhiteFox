
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(16, 16, 7, stride=1, padding=3)
        self.conv2 = torch.nn.Conv2d(16, 16, 7, stride=1, padding=3)
        self.conv3 = torch.nn.Conv2d(16, 16, 7, stride=1, padding=3)
    def forward(self, input1):
        v1 = self.conv1(input1)
        v2 = self.conv2(v1)
        v3 = v1 + v2
        v4 = torch.relu(v3)
        v5 = v1 + input1
        v6 = self.conv3(v5)
        v7 = self.conv1(v6)
        v8 = self.conv2(v7)
        v9 = v1 + v5
        v10 = torch.relu(v9)
        v11 = self.conv3(v10)
        v12 = v10 + v6
        v13 = torch.relu(v12)
        v14 = self.conv2(v13)
        v15 = v6 + input1
        v16 = torch.relu(v15)
        return v16
# Inputs to the model
input1 = torch.randn(1, 16, 64, 64)
