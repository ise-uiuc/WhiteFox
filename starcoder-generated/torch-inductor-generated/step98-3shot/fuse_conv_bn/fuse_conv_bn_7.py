
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv1d(6, 6, 3)
        self.conv2 = torch.nn.Conv1d(5, 5, 3)
        self.conv3 = torch.nn.Conv1d(1, 4, 3)
        self.bn = torch.nn.BatchNorm1d(6)
    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.bn(x1)
        x3 = self.conv2(x2)
        x4 = self.conv3(x3)
        return x4
# Inputs to the model
x = torch.randn(1, 6, 20)
