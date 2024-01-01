
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 16, 5, stride=2, padding=2)
        self.fc = torch.nn.Linear(256, 64)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = v1 - 0.5
        v3 = F.relu(v2)
        v3 = torch.flatten(v3)
        v4 = self.fc(v3)
        return v4
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
