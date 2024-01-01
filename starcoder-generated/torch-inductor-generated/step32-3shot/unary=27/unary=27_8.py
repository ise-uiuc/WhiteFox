
class Model(torch.nn.Module):
    def __init__(self, min, max):
        super().__init__()
        self.min = min
        self.max = max
        self.conv1 = torch.nn.Conv2d(3, 32, 3, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(32, 32, 3, stride=1, padding=1)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v1 = torch.clamp_min(v1, self.min)
        v1 = torch.clamp_max(v1, self.max)
        v1 = F.relu(v1)
        v1 = self.conv2(v1)
        v1 = torch.clamp_min(v1, self.min)
        v1 = torch.clamp_max(v1, self.max)
        v1 = F.relu(v1)
        return v1
min = 0
max = 0.7
# Inputs to the model
x1 = torch.randn(1, 3, 224, 224)
