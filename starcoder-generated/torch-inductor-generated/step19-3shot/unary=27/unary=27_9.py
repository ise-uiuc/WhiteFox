
class Model(torch.nn.Module):
    def __init__(self, min=0.1, max=-0.1):
        super().__init__()
        self.conv = torch.nn.Conv1d(1, 8, 1, stride=2, padding=1)
        self.min = min
        self.max = max
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = torch.clamp_min(v1, self.min)
        v3 = torch.clamp_max(v2, self.max)
        return v3
# Inputs to the model
x1 = torch.randn(1, 1, 128)
