
class Model(torch.nn.Module):
    def __init__(self, min, max):
        super().__init__()
        self.conv0 = torch.nn.Conv2d(1, 1, 3, 1, 0)
        self.relu = torch.nn.ReLU()
        self.conv1 = torch.nn.Conv2d(1, 1, 5, 1, 2)
        self.min = min
        self.max = max
    def forward(self, x1):
        v1 = self.conv0(x1)
        v2 = self.relu(v1)
        v3 = self.conv1(v2)
        v4 = torch.clamp_min(v3, self.min)
        v5 = torch.clamp_max(v4, self.max)
        return v5
min = 0
max = 1
# Inputs to the model
x1 = torch.randn(1, 1, 32, 32)
