
class Model(torch.nn.Module):
    def __init__(self, min, max):
        super().__init__()
        self.conv = torch.nn.Conv1d(2, 4, 5, stride=1, padding=0)
        self.bn = torch.nn.BatchNorm1d(num_features=4, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.max = max
        self.min = min
    def forward(self, x84):
        v1 = self.conv(x84)
        v2 = self.bn(v1)
        v3 = torch.clamp_min(input=v2, min=self.min)
        v4 = torch.clamp_max(input=v3, max=self.max)
        return v4
min = 123
max = 457
# Inputs to the model
x84 = torch.randn(1, 2, 100)
