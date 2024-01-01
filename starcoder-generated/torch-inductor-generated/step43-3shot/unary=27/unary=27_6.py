
class Model(torch.nn.Module):
    def __init__(self, min, max):
        super().__init__()
        self.conv = torch.nn.Conv2d(1, 2, 9, stride=4, padding=1)
        self.fc = torch.nn.Linear(20, 40)
        self.min = min
        self.max = max
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = v1.reshape(v1.size(0), -1)
        v3 = self.fc(v2)
        v4 = torch.clamp_min(v3, self.min)
        v5 = torch.clamp_max(v4, self.max)
        return v5
min = 3.0
max = 4.0
# Inputs to the model
x1 = torch.randn(1, 1, 10, 10)
