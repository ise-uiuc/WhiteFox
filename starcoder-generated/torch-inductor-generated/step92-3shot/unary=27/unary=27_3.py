
class Model(torch.nn.Module):
    def __init__(self, min, max):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 64, 5, stride=2, padding=0.5)
        self.conv2 = torch.nn.Conv2d(64, 64, 5, stride=1, padding=0)
        self.conv3 = torch.nn.Conv2d(64, 1, 5, stride=2, padding=2)
        self.relu = torch.nn.ReLU()
        self.min = min
        self.max = max
    def forward(self, x1):
        v1 = self.relu(self.conv1(x1))
        v2 = self.relu(self.conv2(v1))
        v3 = torch.clamp_min(self.conv3(v2), self.min)
        v4 = torch.clamp_max(v3, self.max)
        return v4
min = 5.0
max = 4.2
# Inputs to the model
x1 = torch.randn(1, 1, 200, 200)
