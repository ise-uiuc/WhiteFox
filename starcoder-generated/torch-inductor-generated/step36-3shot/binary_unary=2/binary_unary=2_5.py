
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(2, 3, 1, stride=2)
        self.conv2 = torch.nn.Conv2d(3, 1, 3, padding=1, stride=1)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = self.conv2(v1)
        v3 = v2 - 0.989
        v4 = F.relu(v3)
        return v4
# Inputs to the model
x1 = torch.randn(1, 2, 64, 64)
