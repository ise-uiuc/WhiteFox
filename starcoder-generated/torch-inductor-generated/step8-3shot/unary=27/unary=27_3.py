
class Model(torch.nn.Module):
    def __init__(self, min_value, max_value, num_features):
        super().__init__()
        self.conv = torch.nn.Conv2d(num_features, num_features, 4, stride=4, padding=4)
        self.min_value = min_value
        self.max_value = max_value
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = torch.clamp_min(v1, self.min_value)
        v3 = torch.clamp_max(v2, self.max_value)
        return v3
min_value = -1
max_value = 2
num_features = 42
# Inputs to the model
x1 = torch.randn(1, num_features, 52, 52)
