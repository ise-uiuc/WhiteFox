
class Model(torch.nn.Module):
    def __init__(self, min, max, num_features):
        super().__init__()
        self.min = min
        self.max = max
        self.conv = torch.nn.Conv2d(num_features, num_features, 3, stride=2, padding=1)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = torch.clamp_min(v1, self.min)
        v3 = torch.clamp_max(v2, self.max)
        return v3
min = 0.2
max = 1.2
num_features = 3
# Inputs to the model
x1 = torch.randn(1, num_features, 224, 224)
