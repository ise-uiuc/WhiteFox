
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 15, 3, stride=3)
        self.conv2 = torch.nn.Conv2d(15, 16, 3, stride=7)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = self.conv2(v1)
        v3 = v2 + 0.1
        return v3
# Inputs to the model
x1 = torch.randn(1, 3, 100, 100)
