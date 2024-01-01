
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv1d(3, 3, 3)
        self.conv2 = torch.nn.Conv1d(3, 3, 3)
        self.bn = torch.nn.BatchNorm1d(3)
        self.relu = torch.nn.ReLU()
    def forward(self, x1):
        s = self.conv1(x1)
        t = self.conv2(s)
        y = self.bn(t)
        z = self.relu(y)
        return s, y, z
# Inputs to the model
x1 = torch.randn(1, 3, 6)
