
class Model(torch.nn.Module):
    def __init__(self, min, max, min_1, max_1):
        super().__init__()
        self.min = min
        self.max = max
        self.conv = torch.nn.Conv3d(1, 7, 7, stride=2, padding=3)
        self.min_1 = min_1
        self.max_1 = max_1
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = torch.clamp_min(v1, self.min)
        v3 = torch.clamp_max(v2, self.max)
        v4 = F.relu(v3)
        v5 = torch.clamp_min(v4, self.min_1)
        v6 = torch.clamp_max(v5, self.max_1)
        return v6
min = 5.9
max = 6.6
min_1 = -3.3
max_1 = 7.5
# Inputs to the model
x1 = torch.randn(1, 1, 80, 80, 80)
