
class Model(torch.nn.Module):
    def __init__(self, min, max):
        super().__init__()
        self.conv11 = torch.nn.Conv2d(16, 16, (1, 1), stride=2, padding=(0, 0))
        self.conv12 = torch.nn.Conv2d(16, 64, (3, 3), stride=1, padding=(1, 1))
        self.conv21 = torch.nn.Conv2d(64, 64, (1, 1), stride=2, padding=(0, 0))
        self.conv31 = torch.nn.Conv2d(64, 128, (1, 1), stride=2, padding=(0, 0))
        self.conv32 = torch.nn.Conv2d(128, 256, (3, 3), stride=1, padding=(2, 2), groups=2)
        self.min = min
        self.max = max
    def forward(self, x1):
        v1 = self.conv11(x1)
        v2 = self.conv12(v1)
        v3 = torch.relu(v2)
        v4 = self.conv21(v3)
        v5 = torch.relu(v4)
        v6 = self.conv31(v5)
        v7 = torch.relu(v6)
        v8 = self.conv32(v7)
        v9 = torch.clamp_min(v8, self.min)
        v10 = torch.clamp_max(v9, self.max)
        return v10
min = 0.3
max = 0.3
# Inputs to the model
x1 = torch.randn(1, 16, 112, 112)
