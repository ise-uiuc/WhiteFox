
class Model(torch.nn.Module):
    def __init__(self, max_value=1, min_value=0):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 3, 1, stride=1, padding=1)
        self.max_value = max_value
        self.max_value_1 = max_value - 1
        self.max_value_2 = max_value - 2
        self.max_value_3 = max_value - 3
        self.min_value = min_value
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = v1 - self.min_value
        v3 = torch.relu(v2)
        v4 = torch.clamp_max(v3, self.max_value_1)
        v5 = torch.clamp_max(v4, self.max_value_2)
        v6 = torch.clamp_max(v5, self.max_value_3)
        v7 = torch.clamp_max(v6, self.max_value)
        return v7
max_value = 1
min_value = 0
# Inputs to the model
x1 = torch.randn(1, 3, 12, 5)
