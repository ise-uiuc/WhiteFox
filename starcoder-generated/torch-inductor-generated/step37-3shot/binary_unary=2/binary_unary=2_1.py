
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 6, 5, stride=1, padding=2)
        self.conv2 = torch.nn.Conv2d(6, 32, 7, stride=2)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = v1 - 3
        v3 = F.relu(v2)
        v4 = self.conv2(v3)
        v5 = v4 - 0.5
        v6 = F.relu(v5)
        v7 = v6.unsqueeze(0)
        return v7
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
