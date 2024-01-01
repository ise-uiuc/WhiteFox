
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv1d(6, 7, 4)
        self.bn = torch.nn.BatchNorm1d(7)
        self.conv2 = torch.nn.Conv1d(7, 8, 2)
    def forward(self, x):
        s = self.conv(x)
        s = self.bn(s)
        t = self.conv2(s)
        return t
# Inputs to the model
x = torch.randn(1, 6, 32)
