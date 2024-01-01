
class Model(torch.nn.Module):
    def __init__(self, min, max):
        super().__init__()
        self.conv0 = torch.nn.Conv2d(1, 1, 1, stride=2, padding=122)
        self.conv1 = torch.nn.Conv2d(1, 1, 1, stride=2, padding=0)
        self.min = min
        self.max = max
    def forward(self, x1):
        v0 = self.conv0(x1)
        v1 = self.conv1(x1)
        v2 = torch.clamp_min(v0, self.min)
        v3 = torch.clamp_max(v2, self.max)
        return v3
min = -0.006424885283130646
max = -0.006424885283130646
# Inputs to the model
x1 = torch.randn(1, 1, 24, 32)
