
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(32, 26, 4, stride=2, padding=3)
    def forward(self, x1):
        v1 = torch.gelu(torch.relu(x1))
        v2 = x1 + 1
        v3 = torch.relu(v1)
        v4 = torch.relu(v2)
        v5 = self.conv1(v3) * v4
        v6 = torch.relu(v5)
        return v6
# Inputs to the model
x1 = torch.randn(1, 32, 128, 128)
