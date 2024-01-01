
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1d1 = torch.nn.Conv1d(32, 64, 1, stride=1)
        self.conv1d2 = torch.nn.Conv1d(64, 128, 1, stride=1)
    def forward(self, x1):
        v1 = self.conv1d1(x1)
        v2 = torch.sigmoid(v1)
        v3 = self.conv1d2(v2)
        v4 = torch.sigmoid(v3)
        return v4
# Inputs to the model
x1 = torch.randn(1, 32, 1024)
