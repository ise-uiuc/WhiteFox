
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.avgpool = torch.nn.AvgPool2d(3, stride=1, padding=1)
        self.fc1 = torch.nn.Linear(32, 16)
        self.fc2 = torch.nn.Linear(16, 32)
    def forward(self, x1):
        t1 = self.avgpool(x1)
        v2 = t1.view(t1.size(0), -1)
        v3 = self.fc1(v2)
        v4 = 3 + v3
        v5 = torch.clamp_min(v4, 0)
        v6 = torch.clamp_max(v5, 6)
        v7 = v3 * v6
        v8 = v6 / 6
        v9 = self.fc2(v8)
        return v9
# Inputs to the model
x1 = torch.randn(2, 3, 64, 64)
# Inputs ends
