
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 3, 5, stride=1, padding=2)
        self.conv2 = torch.nn.Conv2d(3, 3, 5, stride=1, padding=2)
    def forward(self, x1, x2):
        v1 = self.conv1(x1)
        v2 = self.conv2(x1)
        v3 = v1 + v2
        v4 = self.conv1(x2)
        v5 = self.conv2(x2)
        v6 = v4 + v5
        return v3 + v6
# Inputs to the model
x1 = torch.randn(1, 3, 32, 32)
x2 = torch.randn(1, 3, 32, 32)
