
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv1d(3, 2, 3)
        self.bn1 = torch.nn.BatchNorm1d(2)
        self.bn2 = torch.nn.BatchNorm1d(2)
        self.bn3 = torch.nn.BatchNorm1d(2)
    def forward(self, x):
        s = self.conv(x)
        t = self.bn2(s)
        y = self.bn3(s)
        y = t + y
        return x
# Inputs to the model
x = torch.randn(1, 3, 10)
