
class Model(torch.nn.Module):
    def __init__(self, min, max):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 82, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=3, bias=False)
        self.bn = torch.nn.BatchNorm2d(num_features=82)
        self.min = min
        self.max = max
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = self.bn(v1)
        v3 = torch.clamp_min(v2, self.min)
        v4 = torch.clamp_max(v3, self.max)
        return v4
min = 0
max = 2030
# Inputs to the model
x1 = torch.randn(1, 3, 128, 128)
