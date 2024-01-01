
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 16, 1, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(3, 16, 1, stride=1, padding=1)
    def forward(self, x1, other=None):
        v1 = self.conv1(x1)
        self.conv2.weight = torch.nn.Parameter(v1)
        v3 = self.conv2(x1)
        v2 = v3 + other
        return v2
# Inputs to the model
x1 = torch.randn(5, 3, 64, 64)
