
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 32, 1, stride=2, padding=0)
        self.conv2 = torch.nn.Conv2d(32, 32, 1, stride=2, padding=0)
    def forward(self, x):
        v1 = self.conv1(x)
        v2 = self.conv2(v1)
        v3 = v2 - 8.58
        return v3
# Inputs to the model
x = torch.randn(1, 3, 64, 64)
