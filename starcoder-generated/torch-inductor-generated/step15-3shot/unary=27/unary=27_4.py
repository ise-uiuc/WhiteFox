
class Model(torch.nn.Module):
    def __init__(self, min, max):
        super().__init__()
        self.conv = torch.nn.Conv2d(9, 20, 5, stride=1, padding=1)
        self.min = min
        self.max = max
    def forward(self, x1):
        conv = self.conv(x1)
        clamp_min = torch.clamp(conv, min=self.min)
        clamp_max = torch.clamp(clamp_min, max=self.max)
        return clamp_max
min = 0.0
max = 0.0
# Inputs to the model
x1 = torch.randn(1, 9, 32, 32)
