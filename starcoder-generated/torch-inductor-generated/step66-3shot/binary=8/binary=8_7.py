
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
    def forward(self, x):
        v1 = self.conv1(x)
        v2 = F.relu(v1)
        v3 = self.conv2(x)
        v4 = F.relu(v3)
        return v4 + v2
# Inputs to the model
x = torch.randn(1, 3, 64, 64)
