
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv2 = torch.nn.Conv2d(32, 16, 5, stride=2, padding=3)
        self.conv1 = torch.nn.Conv2d(32, 16, 5, stride=2, padding=3)
    def forward(self, x1):
        v1 = self.conv2(x1)
        v2 = self.conv1(x1)
        v1 = v1.add(v2)
        v3 = F.relu(v1)
        return v3
# Inputs to the model
x1 = torch.randn(1, 32, 64, 64)
