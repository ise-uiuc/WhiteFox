
class Model(torch.nn.Module):
    def __init__(self, min, max):
        super().__init__()
        self.conv = torch.nn.Conv2d(1, 1, 1, stride=1, padding=1)
        self.maxpool = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=0)
        self.min = min
        self.max = max
    def forward(self, x1):
        v1 = self.conv(x1)
        v1 = torch.clamp_min(v1, self.min)
        v2 = torch.clamp_max(v1, self.max)
        v1 = self.maxpool(v2)
        return v1
min = -50
max = 49
# Inputs to the model
x1 = torch.randn(2, 1, 10, 10)
