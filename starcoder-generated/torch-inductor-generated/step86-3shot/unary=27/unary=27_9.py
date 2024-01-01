
class Model(torch.nn.Module):
    def __init__(self, min, max):
        super().__init__()
        self.conv = torch.nn.Conv2d(10000, 100000, 1, stride=1, padding=0)
        self.fc = torch.nn.Linear(1, 8)
        self.min = min
        self.max = max
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = torch.clamp_min(v1, self.min)
        v3 = torch.clamp_max(v2, self.max)
        v4 = v3.flatten(start_dim=1)
        v5 = self.fc(v4)
        return v5
min = -1.4465864419364929
max = -0.5888005044937134
# Inputs to the model
x1 = torch.randn(1, 10000, 1, 1)
