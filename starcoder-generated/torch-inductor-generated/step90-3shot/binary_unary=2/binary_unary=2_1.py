
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(9, 24, 5, stride=1, padding=2)
        self.conv2 = torch.nn.Conv2d(7, 14, 1, stride=1, padding=0, bias=False)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = v1 - 0.6
        v3 = F.relu(v2)
        v4 = self.conv2(x1)
        v5 = v4 - 16
        v6 = F.relu(v5)
        return v6
# Inputs to the model
x1 = torch.randn(1, 9, 32, 32)
x2 = torch.randn(1, 7, 64, 64)
