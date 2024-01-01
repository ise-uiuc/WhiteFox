
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(4, 10)
        self.conv = torch.nn.Conv2d(3, 8, 3, stride=1, padding=1)
    def forward(self, x1):
        v1 = self.fc1(x1)
        v2 = v1.reshape([-1, 1, 7, 7])
        v3 = torch.relu(v2)
        v4 = self.conv(v3)
        return v4
# Inputs to the model
x1 = torch.randn(1, 4)
x2 = torch.randn(1, 3, 64, 64)
