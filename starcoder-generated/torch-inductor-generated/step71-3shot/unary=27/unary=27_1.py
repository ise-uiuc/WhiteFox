
class Model(torch.nn.Module):
    def __init__(self, min, max):
        super().__init__()
        self.conv = torch.nn.Conv2d(54, 256, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), bias=False)
        self.bn = torch.nn.BatchNorm2d(num_features=256)
        self.min = min
        self.max = max
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = self.bn(v1)
        v3 = torch.clamp_min(v2, self.min)
        v4 = torch.clamp_max(v3, self.max)
        return v4
min = 4294967294
max = 4294967295
# Inputs to the model
x1 = torch.randn(1, 54, 16, 16)
