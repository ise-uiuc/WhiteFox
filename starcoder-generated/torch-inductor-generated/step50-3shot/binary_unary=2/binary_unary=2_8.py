
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 4, 3, stride=3)
        self.conv2 = torch.nn.Conv2d(4, 10, 2, stride=2)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = v1 - 1
        v3 = F.relu(v2) + 0.5
        v4 = self.conv2(v3)
        v5 = v4 - 0.3
        v6 = F.relu(v5)
        v7 = torch.squeeze(v6, 2)
        return v7
# Inputs to the model
x1 = torch.randn(1, 3, 20, 20)
