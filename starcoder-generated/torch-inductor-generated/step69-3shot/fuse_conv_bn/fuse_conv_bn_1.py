
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(1)
        self.layer = torch.nn.LSTM(16, 32, 1, bidirectional=True)
        torch.manual_seed(1)
        self.bn = torch.nn.BatchNorm2d(32)
    def forward(self, x, h, c):
        y, (h, c) = self.layer(x, (h, c))
        y = self.bn(y)
        y = self.bn(y)
        return y, h, c
# Inputs to the model
x = torch.randn(1, 4, 16, 16)
h = torch.randn(2, 4, 32)
c = torch.randn(2, 4, 32)
# Input to the model ends

# Model begins
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(1)
        self.layer = torch.nn.Linear(16, 32, bias=False)
        torch.manual_seed(1)
        self.bn = torch.nn.BatchNorm1d(32)
    def forward(self, x):
        s = self.layer(x)
        s = self.bn(s)
        return s
# Inputs to the model
x = torch.randn(1, 32, 4, 4)
