
class Model(torch.nn.Module):
    def __init__(self, min, max):
        super().__init__()
        self.conv = torch.nn.Conv1d(1, 1, 1)
        self.min = min
        self.max = max
    def forward(self, x1):
        v0 = torch.clamp_min(self.conv(x1), self.min)
        v1 = torch.clamp_max(v0, self.max)
        return v1
min = 0.1805120863481903
max = -0.9146768007278442
# Inputs to the model
x1 = torch.randn(1, 1, 5)
