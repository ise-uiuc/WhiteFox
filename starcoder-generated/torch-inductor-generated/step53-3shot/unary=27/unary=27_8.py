
class Model(torch.nn.Module):
    def __init__(self, min, max):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(11, 5, 5, stride=1, padding=2)
        self.conv2 = torch.nn.Conv2d(5, 11, 5, stride=1, padding=2)
        self.min = min
        self.max = max
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = torch.clamp_min(v1, self.min)
        v3 = self.conv2(v2)
        return v3
min = -0.9
max = 0.1
# Inputs to the model
x1 = torch.randn(1, 11, 64, 64)
