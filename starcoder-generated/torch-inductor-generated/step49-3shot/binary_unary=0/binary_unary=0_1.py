
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(16, 16, 7, stride=1, padding=3)
        self.conv2 = torch.nn.Conv2d(16, 16, 7, stride=1, padding=3)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = self.conv2(x1)
        v3 = v2 + torch.ones(1, 16, 64, 64)
        v4 = torch.relu(v2)
        v5 = self.conv2(v4)
        v6 = v5 + torch.zeros(1, 16, 64, 64)
        return v6
# Inputs to the model
x1 = torch.randn(1, 16, 64, 64)
