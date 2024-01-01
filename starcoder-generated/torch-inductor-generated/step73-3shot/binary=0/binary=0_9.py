
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(4, 16, 5, stride=1, padding=0)
        self.relu = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv2d(8, 24, 3, stride=1, padding=1)
        self.linear = torch.nn.Linear(24, 8)
    def forward(self, x, other=1):
        v0 = self.conv1(x)
        v1 = self.relu(v0)
        v2 = self.conv2(v1)
        v3 = self.relu(v2)
        v4 = self.linear(v3)
        return v4
# Inputs to the model
x = torch.randn(1, 4, 64, 64)
