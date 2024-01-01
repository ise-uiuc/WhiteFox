
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 50, 5, stride=1, padding=2)
        self.conv2 = torch.nn.Conv2d(10, 80, 5, stride=1, padding=2)
        self.conv3 = torch.nn.Conv2d(100, 100, 5, stride=1, padding=2)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = v1 - 10
        v3 = F.relu(v2)
        v4 = torch.unsqueeze(x1, 1)
        v5 = torch.cat([v3, v4], 1)
        v6 = self.conv2(v5)
        v7 = v6 - torch.round(torch.neg(v6))
        v8 = torch.mean(v7)
        v9 = torch.unsqueeze(v8, 1)
        v10 = torch.cat([v7, v9], 1)
        v11 = self.conv3(v10)
        v12 = v11 - 10
        v13 = F.relu(v12)
        return v13
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
