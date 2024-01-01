
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 8, 3, stride=2, padding=2)
        self.conv2 = torch.nn.Conv2d(8, 16, 3, stride=2, padding=2)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = v1 - 0.5
        v3 = F.relu(v2)
        v4 = v3 - 0.01
        v5 = F.relu(v4)
        v6 = self.conv2(v5)
        v7 = v6 - 0.625
        v8 = F.relu(v7)
        v9 = torch.squeeze(v8, 0)
        return v9
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
