
class Model(torch.nn.Module):
    def __init__(self, min, max):
        super().__init__()
        self.conv = torch.nn.Conv1d(3, 1, 1, stride=1, padding=0)
        self.batchnorm = torch.nn.BatchNorm1d(3)
        self.conv1 = torch.nn.Conv1d(3, 1, 1, stride=1, padding=0)
        self.avgpool = torch.nn.AvgPool1d(1, stride=1, padding=0)
        self.conv2 = torch.nn.Conv1d(3, 1, 1, stride=1, padding=0)
        self.min = min
        self.max = max
    def forward(self, x1):
        v1 = self.conv(x1)
        v1 = self.batchnorm(v1)
        v1 = v1.type(torch.int32)
        v2 = self.conv1(v1)
        v3 = self.avgpool(v2)
        v4 = self.conv2(v3)
        v5 = torch.clamp_min(v4, self.min)
        v6 = torch.clamp_max(v5, self.max)
        return v6
min = 100
max = 10000
# Inputs to the model
x1 = torch.randn(32, 3, 32)
