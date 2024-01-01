
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 1, 3, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(1, 1, 1, stride=1, padding=1)
    def forward(self, x1, other=None):
        v1 = self.conv1(x1)
        v2 = self.conv2(v1)
        if other is None:
            other = v2 + 1
        v3 = v2 + other
        return v3
# Inputs to the model
x1 = torch.randn(1, 1, 100, 100)
