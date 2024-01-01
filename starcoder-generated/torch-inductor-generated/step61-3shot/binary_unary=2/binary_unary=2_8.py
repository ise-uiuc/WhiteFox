
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(64, 128, 1, padding=0)
        self.conv2 = torch.nn.Conv2d(128, 128, 1, padding=0)
        self.fc = torch.nn.Linear(128, 8)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = self.conv2(v1)
        v3 = v2 - 0.3
        v4 = F.relu(v3)
        v5 = self.fc(v4)
        v6 = v5 - torch.floor(v5)
        return v6
# Inputs to the model
x1 = torch.randn(1, 64, 128, 128)
