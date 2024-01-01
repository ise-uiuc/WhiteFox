
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv1d(1, 1, 1)
        self.bn = torch.nn.BatchNorm1d(1)
    def forward(self, x):
        return torch.nn.Sigmoid()(self.bn(self.conv(x)))
# Inputs to the model
x = torch.randn(3, 1)
