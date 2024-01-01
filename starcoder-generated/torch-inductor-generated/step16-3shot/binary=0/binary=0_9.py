
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 3, 3, stride=1, padding=2)
        self.conv2 = torch.nn.Conv2d(3, 3, 3, stride=1, padding=2)
    def forward(self, x1, other=1):
        v1 = self.conv1(x1)
        v2 = self.conv2(v1)
        v3 = v2 + other
        return v3
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
