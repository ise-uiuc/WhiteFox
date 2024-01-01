
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(1)
        self.conv = torch.nn.Conv1d(4, 4, 1)
        torch.manual_seed(1)
        self.bn = torch.nn.BatchNorm1d(4)
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        y = self.conv(x)
        y = self.bn(y)
        y = self.conv(y)
        return y
# Inputs to the model
x = torch.randn(1, 4, 32)
