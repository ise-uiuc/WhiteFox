
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 16, 7, stride=1, padding=3)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v3 = torch.relu(v1)
        v2 = v1 + v3
        v4 = v1 + v2
        v5 = v1 + v3 + v4
        v6 = torch.relu(v5)
        v7 = torch.relu(v5)
        v1001 = torch.add(torch.relu(torch.add(v7, v3)), torch.relu(v6))
        v8 = torch.relu(v6 + v1001)
        return v8
# Inputs to the model
x1 = torch.randn(1, 3, 112, 112)
