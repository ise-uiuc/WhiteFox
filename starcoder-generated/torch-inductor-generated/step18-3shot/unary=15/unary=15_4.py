
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dense1 = torch.nn.Linear(2, 10)
        self.conv1 = torch.nn.Conv2d(3, 10, 1, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(3, 10, 1, stride=3, padding=1)
        self.conv3 = torch.nn.Conv2d(3, 10, 2, stride=1, padding=1)
    def forward(self, x1):
        v1 = self.dense1(x1)
        v2 = F.relu(v1)
        v3 = self.conv1(x1) + v2
        v4 = self.conv2(x1) + torch.max(v2, v3)
        v5 = torch.max(v2, v3, v4)
        v6 = torch.relu(v5)
        v7 = self.conv3(x1) + v6
        v8 = torch.relu(v7)
        v9 = torch.max(v5, v8)
        return v9
# Inputs to the model
x1 = torch.randn(1, 2)
