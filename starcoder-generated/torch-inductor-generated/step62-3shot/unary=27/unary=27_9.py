
class Model(torch.nn.Module):
    def __init__(self, min, max):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 4, 5, stride=1, padding=2)
        self.relu = torch.nn.ReLU()
        self.sigmoid = torch.nn.Sigmoid()
        self.min = min
        self.max = max
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = self.sigmoid(v1)
        v3 = self.relu(v1)
        v4 = self.relu(v3)
        v5 = torch.clamp_min(v4, self.min)
        v6 = self.relu(v5)
        v7 = torch.clamp_max(v6, self.max)
        return v7
min = -0.3
max = 1.8
# Inputs to the model
x1 = torch.randn(1, 3, 20, 20)
