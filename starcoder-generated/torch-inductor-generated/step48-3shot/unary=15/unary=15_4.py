
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
        self.fc1 = torch.nn.Linear(32, 12)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = F.relu(v1)
        v3 = v2.view(-1, 32)
        v4 = self.fc1(v3)
        return v4
# Inputs to the model
x1 = torch.randn(1, 3, 256, 256)
