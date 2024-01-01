
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv1d(2, 2, 2)
        self.bn = torch.nn.BatchNorm1d(2)
    def forward(self, x):
        x = self.conv1(x)
        y = self.bn(x)
        return y
# Inputs to the model
x = torch.randn(1, 2, 8)
