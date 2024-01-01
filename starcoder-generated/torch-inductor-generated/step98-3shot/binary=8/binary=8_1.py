
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(3, 16, 1, stride=1, padding=1)
    def forward(self, x, y):
        v1 = self.conv1(x)
        v2 = self.conv2(y)
        v3 = v1 + v2
        v4 = v3.add(v2)
        return
# Inputs to the model
x = torch.randn(1, 3, 64, 64)
y = torch.randn(1, 3, 64, 64)
