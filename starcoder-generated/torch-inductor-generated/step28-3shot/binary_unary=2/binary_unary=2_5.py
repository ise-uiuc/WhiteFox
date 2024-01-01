
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv1d(3, 32, 3, stride=1, padding=1)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = v1 - 0.3
        v3 = F.relu(v2)
        v4 = self.conv(x1)
        v5 = v1 - 0.1
        v6 = F.relu(v5)
        v7 = v3 + v6
        return v7
# Inputs to the model
x1 = torch.randn(1, 3, 256)
