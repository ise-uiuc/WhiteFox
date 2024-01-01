
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv1d(28, 14, 3)
        self.bn = torch.nn.BatchNorm1d(14)
    def forward(self, x):
        x = self.bn(self.conv(x))
        return x
# Inputs to the model
x = torch.randn(1, 28, 16)
