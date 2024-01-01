
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 8, 3, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(8, 4, 4, stride=1, padding=1)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = torch.mean(v1, 1, True)
        v3 = v2 - 0.8
        v4 = F.relu(v3)
        v5 = self.conv2(v4)
        v6 = F.relu(v5)
        return v6
# Inputs to the model
x1 = torch.randn(1, 3, 784)
