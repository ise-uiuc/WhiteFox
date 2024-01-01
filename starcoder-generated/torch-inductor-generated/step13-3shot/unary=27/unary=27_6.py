
class Model(torch.nn.Module):
    def __init__(self, min_value=1, max_value=-1):
        super().__init__()
        self.fc = torch.nn.Linear(1024, 128)
        self.conv = torch.nn.Conv2d(1, 16, 2, stride=2, padding=1)
        self.min_value = min_value
        self.max_value = max_value
    def forward(self, x1):
        v0 = x1
        v1 = self.fc(v0)

        v2 = self.conv(v1)
        v3 = torch.clamp_min(v2, self.min_value)

        v4 = torch.relu(v2)
        v5 = torch.clamp_max(v4, self.max_value)
        return v5
min = 1
max = -1
# Inputs to the model
x1 = torch.randn(2, 1024)
