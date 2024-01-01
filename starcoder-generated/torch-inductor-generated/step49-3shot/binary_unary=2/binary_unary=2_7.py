
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 64, 3, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(64, 64, 3, stride=1, padding=1)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = v1 - 23
        v3 = F.relu(v2)
        v4 = self.conv2(v3)
        v5 = v4 - 34
        return v5
# Inputs to the model
x1 = torch.randn(1, 1, 32, 32)
