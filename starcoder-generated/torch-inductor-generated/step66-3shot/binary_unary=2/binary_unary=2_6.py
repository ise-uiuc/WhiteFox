
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.relu1 = torch.nn.ReLU()
        self.conv1 = torch.nn.Conv2d(1, 1, 3, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(1, 1, 3, stride=1, padding=1)
    def forward(self, x1):
        v1 = self.relu1(x1)
        v2 = self.conv1(v1)
        v3 = v2 - 0.5
        v4 = F.relu(v3)
        v5 = self.conv2(v4)
        v6 = v5 - 0.5
        v7 = F.relu(v6)
        return v7
# Inputs to the model
x1 = torch.randn(1, 1, 64, 64)
