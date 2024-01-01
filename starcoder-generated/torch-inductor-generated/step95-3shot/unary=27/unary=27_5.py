
class Model(torch.nn.Module):
    def __init__(self, min, max):
        super().__init__()
        self.conv = torch.nn.Conv2d(83, 35, 5, stride=4, padding=3)
        self.min = min
        self.max = max
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = torch.clamp_min(v1, self.min)
        v3 = torch.clamp_max(v2, self.max)
        return v3
min = 0.2584912429654222
max = 1.18402120825
# Inputs to the model
x1 = torch.randn(1, 83, 7, 27)
