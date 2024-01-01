
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 6, 2, stride=2, padding=4)
        self.fc = torch.nn.Linear(832, 1)
    def forward(self, x1):
        v1 = self.conv(x1)
        v1 = v1.flatten(1)
        v2 = self.fc(v1)
        v3 = v2 + 3
        v4 = torch.relu(v3)
        v5 = v4 * 6
        return v5
# Inputs to the model
x1 = torch.randn(1, 3, 28, 28)
