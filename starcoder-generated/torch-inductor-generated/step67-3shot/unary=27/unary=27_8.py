
class Model(torch.nn.Module):
    def __init__(self, min, max):
        super().__init__()
        self.conv = torch.nn.Conv1d(2, 1, kernel_size=(27,), stride=(7,), padding=0, dilation=1, groups=1, bias=True)
        self.min = min
        self.max = max
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = torch.clamp_min(v1, self.min)
        v3 = torch.clamp_max(v2, self.max)
        return v3
min = 1000.0
max = 10000.0
# Inputs to the model
x1 = torch.randn(1, 2, 4325)
