
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(1)
        self.conv1 = torch.nn.Conv1d(1, 1, 3)
        torch.manual_seed(1)
        self.bn = torch.nn.BatchNorm1d(1)
    def forward(self, x):
        x1 = self.conv1(x)
        y1 = self.bn(x1)
        x1 = self.conv1(x1)
        y1 = self.bn(x1)
        return y1
# Inputs to the model
x = torch.randn(1, 1, 4, 4)
