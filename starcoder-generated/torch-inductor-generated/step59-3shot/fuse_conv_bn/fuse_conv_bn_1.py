
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv1d(1, 1, 2)
        self.bn = torch.nn.BatchNorm1d(1)
        self.pool2d = torch.nn.MaxPool1d(2)
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.pool2d(x)
        return x
# Inputs to the model
x = torch.randn(1, 1, 12)
