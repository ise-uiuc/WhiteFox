
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv1d(3, 3, 3)
        self.bn = torch.nn.BatchNorm1d(3)
    def forward(self, x):
        l1 = self.conv(x) + 32
        l2 = self.bn(l1) * 0.37
        return l2
# Inputs to the model
x = torch.randn(1, 3, 6)
