
class Model(torch.nn.Module):
    def __init__(self, min, max):
        super().__init__()
        self.conv = torch.nn.Conv2d(43, 9, 4, stride=1, padding=2)
        self.min = min
        self.max = max
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = torch.clamp_min(v1, self.min)
        v3 = torch.clamp_max(v2, self.max)
        return v3
min = 1.4
max = 1.9
# Inputs to the model
x1 = torch.rand(1, 43, 570, 800)
x1 = torch.rand(batch_size, 16, 8, 18)
