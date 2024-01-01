
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 16, 3, stride=1, padding=1)
    def forward(self, x1):
        v1 = self.conv1(x1) #
        v2 = self.conv1(x1)
        v3 = self.conv1(x1)
        v4 = self.conv1(x1)
        v5 = v1 + v2
        v6 = torch.relu(v3)
        v7 = v4 + v5
        v8 = torch.relu(v7)
        v9 = self.conv1(x1) #
        v10 = self.conv1(x1)
        v11 = self.conv1(x1)
        v12 = self.conv1(x1)
        v13 = v9 + v10
        v14 = torch.relu(v11)
        v15 = v12 + v13
        v16 = torch.relu(v15)
        v17 = v8 + v14 + v16
        return v17
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
