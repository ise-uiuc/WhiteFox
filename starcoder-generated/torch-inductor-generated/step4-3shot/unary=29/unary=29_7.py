
class Model(torch.nn.Module):
    def __init__(self, min_value=1.165433630886078, max_value=9.495014390253015):
        super().__init__()
        self.linear = torch.nn.Linear(12, 8)
        self.max_value = max_value
        self.min_value = min_value
        self.relu = torch.nn.ReLU()
        self.t = torch.nn.Conv3d(3, 8, 1, stride=1, padding=1)
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = self.relu(v1)
        v3 = self.t(v2)
        v4 = torch.clamp(v3, self.min_value, self.max_value)
        v5 = self.relu(v1)
        return v4
# Inputs to the model
x1 = torch.randn(1, 12)
