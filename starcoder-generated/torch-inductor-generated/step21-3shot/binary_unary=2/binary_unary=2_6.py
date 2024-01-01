
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(64, 8, 4, stride=2, padding=1)
        self.conv2 = torch.nn.Conv2d(64, 8, 3, stride=2, padding=1)
        self.conv3 = torch.nn.Conv2d(8, 8, 3, stride=1, padding=0)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = F.gelu(v1)
        v3 = v2 - 0.2
        v4 = F.relu(v3)
        v5 = self.conv2(v4)
        v6 = F.gelu(v5)
        v7 = v6 - 1
        v8 = F.relu(v7)
        v9 = self.conv3(v8)
        v10 = v9 - 0.5
        v11 = F.relu(v10)
        return v11
# Inputs to the model
x1 = torch.randn(1, 64, 64, 64)
