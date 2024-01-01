
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 8, 5, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(3, 8, 3, stride=1, padding=1)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = self.conv2(x1)
        v1 = 1 * v1
        v2 = 1 * v2
        v = v1 - v2
        v3 = v[0]
        v4 = v + 0.5
        v5 = F.relu(v3 + v4)
        return v5
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
