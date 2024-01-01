
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 8, 1, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(2, 4, 3, stride=2, padding=1)
    def forward(self, x):
        v1 = self.conv1(x)
        v3 = self.conv2(v1)
        v2 = v3 - 0.0
        return v2
# Inputs to the model
x = torch.randn(1, 1, 64, 64)
