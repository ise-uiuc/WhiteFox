
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 8, 3, stride=1, padding=1)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = v1 - 0.5
        v3 = torch.relu(v2)
        v4 = v2 - 0.25
        v5 = torch.relu(v4)
        v6 = v2 - 0.1
        v7 = torch.relu(v6)
        v8 = v2 - 0.05
        v9 = torch.tanh(x1)
        return v6
# Inputs to the model
x1 = torch.randn(1, 3, 32, 32)
