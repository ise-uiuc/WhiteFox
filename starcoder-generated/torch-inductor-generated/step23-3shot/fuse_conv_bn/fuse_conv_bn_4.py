
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv1d(7, 7, 7)
        self.bn = torch.nn.BatchNorm1d(7)
    def forward(self, x, y):
        z = self.conv1(x)
        s = self.conv1(y)
        t = self.bn(z)
        u = self.bn(s)
        return torch.cat((t, u), 1)
# Inputs to the model
x = torch.randn(1, 7, 10)
y = torch.randn(1, 7, 10)
