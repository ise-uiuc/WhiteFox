
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv1d(3, 3, 1)
        self.bn = torch.nn.BatchNorm1d(3)
    def forward(self, x):
        x = self.conv(x)
        y = self.bn(x)
        return y, x
# Inputs to the model
x = torch.randn(1, 3, 3)
