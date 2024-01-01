
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 8, 5, stride=2, padding=1)
        self.fc = torch.nn.Linear(408, 8)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = F.relu(v1)
        v3 = torch.flatten(v2, 1)
        v4 = self.fc(v3)
        v5 = F.relu(v4)
        return v5
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
