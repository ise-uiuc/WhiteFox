
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv1d(1, 1, 1)
        self.bn = torch.nn.BatchNorm1d(1)
        self.pool1d = torch.nn.MaxPool1d(3)
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.pool1d(x)
        return x
# Inputs to the model
x = torch.randn(1, 1, 1)
