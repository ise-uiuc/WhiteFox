
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 3, 3, stride=2, padding=2)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = F.relu(v1 - 0.5)
        v3 = v2 + 127
        v4 = F.relu(v3)
        return v4
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
