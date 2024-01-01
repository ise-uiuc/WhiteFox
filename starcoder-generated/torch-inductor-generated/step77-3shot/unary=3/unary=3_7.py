
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(1, 9, 2, stride=1, padding=1)
        self.fc = torch.nn.Linear(105, 200)
        self.fc2 = torch.nn.Linear(200, 3)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = v1.flatten(start_dim=1, end_dim=3)
        v3 = self.fc(v2)
        v4 = v3 * 0.7071067811865476
        v5 = torch.erf(v4)
        v6 = v5 + 1
        v7 = v3 * v6
        v8 = self.fc2(v7)
        return v8
# Inputs to the model
x1 = torch.randn(1, 1, 22, 33)
