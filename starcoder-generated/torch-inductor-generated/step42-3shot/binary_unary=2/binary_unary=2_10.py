
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 32, 3, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(32, 16, 3, stride=1, padding=1)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = v1 - 1
        v3 = F.relu(v2)
        v4 = self.conv2(v3)
        v5 = v4 - 2
        v6 = F.relu(v5)
        return v6
# Inputs to the model
x1 = torch.randn(1, 3, 255, 255)
