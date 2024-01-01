
class Model(torch.nn.Module):
    def __init__(self, min, max):
        super().__init__()
        self.conv = torch.nn.Conv2d(16, 8, 3, stride=2, padding=1)
        self.bn = torch.nn.BatchNorm2d(8, eps=0.0003, momentum=0.99)
        self.min = min
        self.max = max
    def forward(self, x1):
        conv = self.conv(x1)
        bn = self.bn(conv)
        clamp_min = torch.clamp(bn, min=self.min)
        clamp_max = torch.clamp(clamp_min, max=self.max)
        return clamp_max
min = 1.0
max = 1.5
# Inputs to the model
x1 = torch.tensor([-1.1, 0.8]).reshape(-1, 1, 1, 2)
