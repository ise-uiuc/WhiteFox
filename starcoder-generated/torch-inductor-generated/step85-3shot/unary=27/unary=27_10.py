
class Model(torch.nn.Module):
    def __init__(self, min, max):
        super().__init__()
        self.min = min
        self.max = max
        self.conv = torch.nn.Conv2d(10, 10, 3, stride=2, padding=4)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = torch.clamp_min(v1, self.min)
        v3 = torch.clamp_max(v2, self.max)
        return v3
min = 10.84561824798584
max = 249.23599243164062
# Inputs to the model
x1 = torch.randn(1, 10, 1024, 256)
