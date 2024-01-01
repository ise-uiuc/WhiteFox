
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv1d(4, 4, 3)
        self.relu = torch.nn.ReLU()
        self.bn = torch.nn.BatchNorm1d(4)
    def forward(self, x):
        x2 = self.conv(x)
        x2 = self.relu(x)
        x2 = self.bn(x2)
        return x2
# Inputs to the model
x = torch.randn(1, 4, 3)
