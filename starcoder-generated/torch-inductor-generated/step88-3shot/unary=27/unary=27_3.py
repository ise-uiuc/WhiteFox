
class Model(torch.nn.Module):
    def __init__(self, max):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 32, 16, stride=2, padding=3)
        self.conv2 = torch.nn.Conv2d(32, 16, 2, stride=2, padding=1)
        self.max = max
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = self.conv2(v1)
        v3 = torch.clamp_max(v2, self.max)
        return v3
max = 1.7
# Inputs to the model
x1 = torch.randn(1, 3, 224, 224)
