
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv1d(1, 1, 2)
        self.bn = torch.nn.BatchNorm1d(1)
    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.bn(x1)
        return x2
# Inputs to the model
x = torch.randn(1, 1, 4)
