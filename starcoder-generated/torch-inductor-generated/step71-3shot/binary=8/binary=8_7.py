
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 8, 1, stride=2, padding=1)
        self.bn1 = lambda x: x + 5.0
    def forward(self, x1, x2):
        v1 = self.conv1(x1)
        v2 = self.conv1(x2)
        v3 = self.bn1(v1)
        v4 = F.softmax(v2)
        return v3 + v4
# Inputs to the model
x1 = torch.randn(1, 3, 16, 16)
x2 = torch.randn(1, 3, 16, 16)
