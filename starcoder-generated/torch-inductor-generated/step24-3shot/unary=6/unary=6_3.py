
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(64, 384, 2, stride=2, padding=5)
        self.relu6 = torch.nn.ReLU6()
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = torch.add(v1, 3)
        v3 = self.relu6(v2)
        v5 = v3 / 6
        return v5
# Inputs to the model
x1 = torch.randn(2, 64, 28, 28)
