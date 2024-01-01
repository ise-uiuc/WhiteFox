
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = nn.Conv1d(3, 64, 3)
        self.bn1 = nn.BatchNorm1d(64)
        self.conv2 = nn.Conv1d(64, 64, 1)
        self.bn2 = nn.BatchNorm1d(64)
        self.sig = nn.Sigmoid()
    def forward(self, x):
        y = self.conv1(x)
        y = self.bn1(y)
        y = self.conv2(y)
        y = self.bn2(y)
        return self.sig(y)
# Inputs to the model
x = torch.randn(1, 3, 4)
