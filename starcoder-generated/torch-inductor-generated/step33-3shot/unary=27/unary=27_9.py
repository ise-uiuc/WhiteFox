
class Model(torch.nn.Module):
    def __init__(self, min, max):
        super().__init__()
        self.conv = torch.nn.Conv1d(64, 8192, 1, bias=False)
        self.max = max
        self.min = min
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = torch.clamp_min(v1, self.min)
        v3 = torch.clamp_max(v2, self.max)
        return v3
min = 0.840443
max = 1.39427
# Inputs to the model
x1 = torch.randn(1024, 64, 7)
