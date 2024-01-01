
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 8, 1, stride=1, padding=0)
        self.fc = torch.nn.Linear(64, 1)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = v1.view(-1, 64)
        v3 = self.fc(v2)
        v4 = torch.matmul(-0.005, v1)
        return v3 + v4
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
