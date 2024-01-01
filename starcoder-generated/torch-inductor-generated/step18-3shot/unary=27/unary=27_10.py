
class Model(nn.Module):
    def __init__(self, min, max):
        super().__init__()
        self.padding = 2
        self.conv = nn.Conv2d(4, 2, 9, stride=2, padding=self.padding)
        self.min = min
        self.max = max
    def forward(self, inputs):
        v1 = self.conv(inputs)
        v2 = torch.clamp_min(v1, self.min)
        v3 = torch.clamp_max(v2, self.max)
        return v3
min = 0.5
max = 0.7
inputs = torch.Tensor(1, 4, 15, 15)
